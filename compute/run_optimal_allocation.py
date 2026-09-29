import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openfermion as of
from ogn.loaders import HDF5Loader
from ogn.hamiltonian import Hamiltonian
from ogn.sorted_insertion import sorted_insertion_grouping
from ogn.adhoc_repacking import adhoc_repacking
from ogn.posthoc_repacking import posthoc_repacking
from ogn.covariance_jit import compute_variance_with_covariance

import quimb.tensor as qtn
from ogn.tensor_utils import pauli_sum_to_mpo, mpo_mps_expectation

import cirq

MOLECULE_DIR = Path(__file__).parent.parent / 'data' / 'molecules'
MOLECULE_PATHS = {name: f'{name}.hdf5' for name in ['LiH', 'H6', 'BeH2', 'H2O', 'NH3', 'CH4']}
METHODS = ['sorted_insertion', 'adhoc_repacking', 'adhoc_si_allocation', 'posthoc_repacking']

TOTAL_SHOTS = 100_000
DMRG_CHI = 64
RANDOM_SEED = 42
MPO_MAX_BOND = 100
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'results_optimal_cccbdb'


def resolve_molecule_path(name):
    path = MOLECULE_DIR / MOLECULE_PATHS[name]
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Cannot find {path}; run compute/generate_molecules.py first")


def build_hamiltonian_mpo(hamiltonian, n_qubits, qubits, max_bond=100):
    qubop = of.QubitOperator()
    for p in hamiltonian.terms:
        term = []
        for i in range(n_qubits):
            x_bit = (p.x_bits >> i) & 1
            z_bit = (p.z_bits >> i) & 1
            if x_bit and z_bit:
                term.append((i, 'Y'))
            elif x_bit:
                term.append((i, 'X'))
            elif z_bit:
                term.append((i, 'Z'))
        qubop += of.QubitOperator(tuple(term) if term else (), float(np.real(p.coeff)))
    psum = of.transforms.qubit_operator_to_pauli_sum(qubop)
    return pauli_sum_to_mpo(psum, qubits, max_bond)


def compute_exp_mps(x_bits, z_bits, n_qubits, qubits, mps, max_bond, cache):
    key = (x_bits, z_bits)
    if key in cache:
        return cache[key]
    if x_bits == 0 and z_bits == 0:
        cache[key] = 1.0
        return 1.0
    term = []
    for i in range(n_qubits):
        xb = (x_bits >> i) & 1
        zb = (z_bits >> i) & 1
        if xb and zb:
            term.append((i, 'Y'))
        elif xb:
            term.append((i, 'X'))
        elif zb:
            term.append((i, 'Z'))
    qubop = of.QubitOperator(tuple(term) if term else (), 1.0)
    psum = of.transforms.qubit_operator_to_pauli_sum(qubop)
    mpo = pauli_sum_to_mpo(psum, qubits, max_bond)
    val = float(np.real(mpo_mps_expectation(mpo, mps)))
    cache[key] = val
    return val


def compute_si_sigma_state_independent(groups):
    sigmas = []
    for group in groups.groups:
        var_g = sum(float(np.real(p.coeff))**2
                    for p in group.paulis if not (p.x_bits == 0 and p.z_bits == 0))
        sigmas.append(np.sqrt(max(0.0, var_g)))
    return sigmas


def compute_optimal_allocation(sigmas, total_shots):
    s = sum(sigmas)
    if s < 1e-12:
        return [total_shots / len(sigmas)] * len(sigmas)
    return [total_shots * sig / s for sig in sigmas]


def variance_product_state(groups, exp_X, exp_Y, exp_Z, n_qubits, shots):
    Ni_map = {}
    for g_idx, group in enumerate(groups.groups):
        for p in group.paulis:
            key = (p.x_bits, p.z_bits)
            Ni_map[key] = Ni_map.get(key, 0.0) + shots[g_idx]

    n_groups = len(groups.groups)
    total_m = sum(len(g.paulis) for g in groups.groups)
    gx = np.zeros(total_m, dtype=np.int64)
    gz = np.zeros(total_m, dtype=np.int64)
    gc = np.zeros(total_m, dtype=np.float64)
    ge = np.zeros(total_m, dtype=np.float64)
    gNi = np.zeros(total_m, dtype=np.float64)
    gs = np.zeros(n_groups, dtype=np.int64)
    gn = np.zeros(n_groups, dtype=np.int64)
    offset = 0
    for g_idx, group in enumerate(groups.groups):
        gs[g_idx] = offset
        gn[g_idx] = len(group.paulis)
        for p in group.paulis:
            gx[offset] = p.x_bits
            gz[offset] = p.z_bits
            gc[offset] = float(np.real(p.coeff))
            x, z = p.x_bits, p.z_bits
            val = 1.0
            for q in range(n_qubits):
                xq = (x >> q) & 1
                zq = (z >> q) & 1
                if xq and zq:
                    val *= exp_Y[q]
                elif xq:
                    val *= exp_X[q]
                elif zq:
                    val *= exp_Z[q]
            ge[offset] = val
            gNi[offset] = Ni_map[(p.x_bits, p.z_bits)]
            offset += 1
    shots_arr = np.array(shots, dtype=np.float64)
    return compute_variance_with_covariance(
        gx, gz, gc, ge, gNi, gs, gn, shots_arr, exp_X, exp_Y, exp_Z, n_groups, n_qubits)


def variance_mps(groups, n_qubits, qubits, mps, shots, max_bond=64, cache=None):
    Ni_map = {}
    for g_idx, group in enumerate(groups.groups):
        for p in group.paulis:
            key = (p.x_bits, p.z_bits)
            Ni_map[key] = Ni_map.get(key, 0.0) + shots[g_idx]

    cache = {} if cache is None else cache
    total_var = 0.0
    _PHASES = [[0,0,0,0],[0,0,1,3],[0,3,0,1],[0,1,3,0]]

    for g_idx, group in enumerate(groups.groups):
        ng = shots[g_idx]
        if ng <= 0:
            continue
        paulis = group.paulis

        for p in paulis:
            e = compute_exp_mps(p.x_bits, p.z_bits, n_qubits, qubits, mps, max_bond, cache)
            c = float(np.real(p.coeff))
            Ni = Ni_map[(p.x_bits, p.z_bits)]
            total_var += c * c * ng / (Ni * Ni) * (1.0 - e * e)

        for i in range(len(paulis)):
            pi = paulis[i]
            ci = float(np.real(pi.coeff))
            Ni = Ni_map[(pi.x_bits, pi.z_bits)]
            ei = cache[(pi.x_bits, pi.z_bits)]
            for j in range(i + 1, len(paulis)):
                pj = paulis[j]
                cj = float(np.real(pj.coeff))
                Nj = Ni_map[(pj.x_bits, pj.z_bits)]
                ej = cache[(pj.x_bits, pj.z_bits)]

                prod_x = pi.x_bits ^ pj.x_bits
                prod_z = pi.z_bits ^ pj.z_bits
                power = 0
                for q in range(n_qubits):
                    p1 = 2 * ((pi.z_bits >> q) & 1) + ((pi.x_bits >> q) & 1)
                    p2 = 2 * ((pj.z_bits >> q) & 1) + ((pj.x_bits >> q) & 1)
                    power += _PHASES[p1][p2]
                phase_sign = -1.0 if power % 4 == 2 else 1.0

                ek = compute_exp_mps(prod_x, prod_z, n_qubits, qubits, mps, max_bond, cache)
                cov = phase_sign * ek - ei * ej
                total_var += 2.0 * ci * cj * ng / (Ni * Nj) * cov

    return total_var


_WORKER = {}


def _init_exp_worker(mps, n_qubits, max_bond):
    _WORKER.update(mps=mps, n_qubits=n_qubits, qubits=cirq.LineQubit.range(n_qubits), max_bond=max_bond)


def _exp_chunk(keys):
    w, cache = _WORKER, {}
    return [(k, compute_exp_mps(k[0], k[1], w['n_qubits'], w['qubits'], w['mps'], w['max_bond'], cache))
            for k in keys]


def needed_pauli_keys(groups):
    keys = set()
    for group in groups.groups:
        paulis = group.paulis
        for i, pi in enumerate(paulis):
            keys.add((pi.x_bits, pi.z_bits))
            for pj in paulis[i + 1:]:
                keys.add((pi.x_bits ^ pj.x_bits, pi.z_bits ^ pj.z_bits))
    return keys


def prefill_expectations(keys, mps, n_qubits, workers, max_bond=64):
    from concurrent.futures import ProcessPoolExecutor
    todo = sorted(keys)
    n_chunks = workers * 8
    chunks = [todo[i::n_chunks] for i in range(n_chunks)]
    cache = {}
    with ProcessPoolExecutor(workers, initializer=_init_exp_worker,
                             initargs=(mps, n_qubits, max_bond)) as ex:
        for part in ex.map(_exp_chunk, chunks):
            cache.update(part)
    return cache


TIE_DECIMALS = 10


def permute_ties(hamiltonian, rng):
    terms = hamiltonian.terms
    keys = rng.random(len(terms))
    order = sorted(range(len(terms)), key=lambda i: (-round(abs(terms[i].coeff), TIE_DECIMALS), keys[i]))
    return Hamiltonian([terms[i] for i in order], hamiltonian.metadata)


def run_tie_samples(args, hamiltonian, n_qubits, qubits, state, dmrg_energy, prov):
    exp_X, exp_Y, exp_Z, mps = state
    samples = []
    for k in range(args.tie_samples):
        ham = permute_ties(hamiltonian, np.random.default_rng([args.seed, k]))
        base = sorted_insertion_grouping(ham, sort_descending=False)
        adhoc = adhoc_repacking(ham, base)
        si_shots = compute_optimal_allocation(compute_si_sigma_state_independent(base), TOTAL_SHOTS)
        opt_shots, _ = adhoc.shot_count_optimized(TOTAL_SHOTS, warm_start=np.array(si_shots))
        samples.append({
            'sorted_insertion': (base, si_shots), 'adhoc_si_allocation': (adhoc, si_shots),
            'adhoc_repacking': (adhoc, list(opt_shots)),
            'posthoc_repacking': (posthoc_repacking(ham, base), si_shots),
        })

    cache = None
    if mps is not None and args.workers > 1:
        keys = set().union(*(needed_pauli_keys(g) for s in samples for g, _ in s.values()))
        cache = prefill_expectations(keys, mps, n_qubits, args.workers)
        print(f"  Prefilled {len(cache)} expectation values on {args.workers} workers", flush=True)

    records = []
    for k, s in enumerate(samples):
        rec = {'seed': [args.seed, k], 'num_groups': s['sorted_insertion'][0].num_groups()}
        for method, (groups, shots) in s.items():
            if mps is not None:
                rec[method] = to_real(variance_mps(groups, n_qubits, qubits, mps, shots, cache=cache))
            else:
                rec[method] = to_real(variance_product_state(groups, exp_X, exp_Y, exp_Z, n_qubits, shots))
        records.append(rec)
        print(f"  sample {k}: groups {rec['num_groups']}  " + "  ".join(
            f"{m} {rec['sorted_insertion'] / rec[m]:.3f}" for m in METHODS[1:]), flush=True)

    result = {
        'molecule': args.molecule, 'state_type': args.state_type, 'n_qubits': n_qubits,
        'n_terms': hamiltonian.num_terms(), 'dmrg_energy': dmrg_energy,
        'tie_decimals': TIE_DECIMALS, 'tie_samples': args.tie_samples, 'seed': args.seed,
        'samples': records, 'provenance': prov,
    }
    fname = f"{args.molecule}_{args.state_type}_tiesamples.json"
    with open(OUTPUT_DIR / fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {fname}")


def to_real(x):
    if isinstance(x, (complex, np.complexfloating)):
        return float(x.real)
    elif isinstance(x, np.ndarray):
        return [to_real(v) for v in x]
    elif isinstance(x, (np.integer, np.floating)):
        return float(x)
    elif isinstance(x, list):
        return [to_real(v) for v in x]
    return x


def provenance(mol_path):
    import hashlib, platform, subprocess, pyscf, openfermionpyscf, quimb, numba
    repo = Path(__file__).parent.parent
    try:
        commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'],
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    return {
        'git_commit': commit, 'git_dirty': dirty,
        'molecule_file': Path(mol_path).name,
        'molecule_sha256': hashlib.sha256(Path(mol_path).read_bytes()).hexdigest(),
        'host': platform.node(), 'python': platform.python_version(),
        'versions': {'numpy': np.__version__, 'openfermion': of.__version__, 'pyscf': pyscf.__version__,
                     'openfermionpyscf': openfermionpyscf.__version__, 'quimb': quimb.__version__,
                     'cirq': cirq.__version__, 'numba': numba.__version__},
        'total_shots': TOTAL_SHOTS, 'dmrg_chi': DMRG_CHI, 'mpo_max_bond': MPO_MAX_BOND,
        'random_state_seed': RANDOM_SEED,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('molecule', choices=list(MOLECULE_PATHS.keys()))
    parser.add_argument('state_type', choices=['dmrg', 'hf', 'random'])
    parser.add_argument('method', choices=METHODS + ['all'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--load-mps', type=str, default=None)
    parser.add_argument('--tie-samples', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    methods = METHODS if args.method == 'all' else [args.method]

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    mol_path = resolve_molecule_path(args.molecule)
    hamiltonian = HDF5Loader(mol_path, hermitianize=True).load()

    n_qubits = hamiltonian.num_qubits()
    n_electrons = hamiltonian.metadata.get('n_electrons', n_qubits // 2)
    qubits = cirq.LineQubit.range(n_qubits)

    print(f"{args.molecule} | {args.state_type} | {args.method} | {n_qubits}q {n_electrons}e {hamiltonian.num_terms()}t")

    baseline = sorted_insertion_grouping(hamiltonian)
    groupings = {}
    for method in methods:
        if method == 'sorted_insertion':
            groupings[method] = baseline
        elif method in ('adhoc_repacking', 'adhoc_si_allocation'):
            groupings[method] = adhoc_repacking(hamiltonian, baseline)
        elif method == 'posthoc_repacking':
            groupings[method] = posthoc_repacking(hamiltonian, baseline)

    dmrg_energy = None
    mps = None
    exp_X = exp_Y = exp_Z = None
    cache = None

    if args.state_type == 'dmrg':
        import pickle
        if args.load_mps:
            with open(args.load_mps, 'rb') as f:
                mps, dmrg_energy = pickle.load(f)
        else:
            ham_mpo = build_hamiltonian_mpo(hamiltonian, n_qubits, qubits, MPO_MAX_BOND)
            dmrg = qtn.DMRG2(ham_mpo, bond_dims=[DMRG_CHI])
            dmrg.solve(tol=1e-6, verbosity=0)
            mps = dmrg.state
            dmrg_energy = float(np.real(dmrg.energy))
            with open(OUTPUT_DIR / f"{args.molecule}_dmrg_chi{DMRG_CHI}_mps.pkl", 'wb') as f:
                pickle.dump((mps, dmrg_energy), f)
        print(f"  DMRG energy: {dmrg_energy:.8f}", flush=True)
        if args.workers > 1 and args.tie_samples == 0:
            keys = set().union(*(needed_pauli_keys(g) for g in groupings.values()))
            cache = prefill_expectations(keys, mps, n_qubits, args.workers)
            print(f"  Prefilled {len(cache)} expectation values on {args.workers} workers", flush=True)
    elif args.state_type == 'hf':
        exp_X = np.zeros(n_qubits)
        exp_Y = np.zeros(n_qubits)
        exp_Z = np.array([-1.0 if i < n_electrons else 1.0 for i in range(n_qubits)])
    elif args.state_type == 'random':
        rng = np.random.RandomState(RANDOM_SEED)
        thetas = rng.uniform(0, np.pi, size=n_qubits)
        phis = rng.uniform(0, 2 * np.pi, size=n_qubits)
        exp_X = np.sin(2 * thetas) * np.cos(phis)
        exp_Y = np.sin(2 * thetas) * np.sin(phis)
        exp_Z = np.cos(2 * thetas)

    si_sigmas = compute_si_sigma_state_independent(baseline)
    prov = provenance(mol_path)

    if args.tie_samples > 0:
        run_tie_samples(args, hamiltonian, n_qubits, qubits, (exp_X, exp_Y, exp_Z, mps), dmrg_energy, prov)
        return

    for method in methods:
        groups = groupings[method]
        if method == 'sorted_insertion':
            shots = compute_optimal_allocation(si_sigmas, TOTAL_SHOTS)
            alloc_type = 'si_optimal'
        elif method in ('posthoc_repacking', 'adhoc_si_allocation'):
            shots = compute_optimal_allocation(si_sigmas, TOTAL_SHOTS)
            alloc_type = 'si_optimal'
        elif method == 'adhoc_repacking':
            warm = np.array(compute_optimal_allocation(si_sigmas, TOTAL_SHOTS))
            shots, _ = groups.shot_count_optimized(TOTAL_SHOTS, warm_start=warm)
            shots = list(shots)
            alloc_type = 'optimized'

        if args.state_type == 'dmrg':
            var = variance_mps(groups, n_qubits, qubits, mps, shots, cache=cache)
        else:
            var = variance_product_state(groups, exp_X, exp_Y, exp_Z, n_qubits, shots)

        print(f"  {method}: variance {var:.6e}")

        result = {
            'molecule': args.molecule, 'method': method, 'state_type': args.state_type,
            'allocation_type': alloc_type, 'n_qubits': n_qubits, 'n_electrons': n_electrons,
            'n_terms': hamiltonian.num_terms(), 'num_groups': groups.num_groups(),
            'total_variance': to_real(var), 'dmrg_energy': dmrg_energy,
            'provenance': prov,
        }
        fname = f"{args.molecule}_{args.state_type}_{method}_optimal_result.json"
        with open(OUTPUT_DIR / fname, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {fname}")


if __name__ == "__main__":
    main()
