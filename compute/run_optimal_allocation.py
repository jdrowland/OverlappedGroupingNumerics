import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openfermion as of
from ogn.loaders import HDF5Loader
from ogn.sorted_insertion import sorted_insertion_grouping
from ogn.adhoc_repacking import adhoc_repacking
from ogn.posthoc_repacking import posthoc_repacking
from ogn.covariance_jit import compute_variance_with_covariance

import quimb.tensor as qtn
from ogn.tensor_utils import pauli_sum_to_mpo, mpo_mps_expectation

import cirq

MOLECULE_PATHS = {
    'LiH': 'H1-Li1_sto-3g_singlet_LiH.hdf5',
    'H6': 'H6_sto-3g_singlet_H6.hdf5',
    'BeH2': 'H2-Be1_sto-3g_singlet_BeH2.hdf5',
    'H2O': 'H2-O1_sto-3g_singlet_H2O.hdf5',
    'NH3': 'H3-N1_sto-3g_singlet_NH3.hdf5',
    'CH4': 'H4-C1_sto-3g_singlet_CH4.hdf5',
}

TOTAL_SHOTS = 100_000
DMRG_CHI = 64
RANDOM_SEED = 42
MPO_MAX_BOND = 100
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'results_optimal'


def resolve_molecule_path(name):
    fname = MOLECULE_PATHS[name]
    of_data = Path(of.__file__).parent / 'testing' / 'data' / fname
    if of_data.exists():
        return str(of_data)
    raise FileNotFoundError(f"Cannot find {fname} in openfermion data dir")


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


def variance_mps(groups, n_qubits, qubits, mps, shots, max_bond=64):
    Ni_map = {}
    for g_idx, group in enumerate(groups.groups):
        for p in group.paulis:
            key = (p.x_bits, p.z_bits)
            Ni_map[key] = Ni_map.get(key, 0.0) + shots[g_idx]

    cache = {}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('molecule', choices=list(MOLECULE_PATHS.keys()))
    parser.add_argument('state_type', choices=['dmrg', 'hf', 'random'])
    parser.add_argument('method', choices=['sorted_insertion', 'adhoc_repacking',
                                           'adhoc_si_allocation', 'posthoc_repacking'])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    mol_path = resolve_molecule_path(args.molecule)
    hamiltonian = HDF5Loader(mol_path, hermitianize=True).load()

    n_qubits = hamiltonian.num_qubits()
    n_electrons = hamiltonian.metadata.get('n_electrons', n_qubits // 2)
    qubits = cirq.LineQubit.range(n_qubits)

    print(f"{args.molecule} | {args.state_type} | {args.method} | {n_qubits}q {n_electrons}e {hamiltonian.num_terms()}t")

    baseline = sorted_insertion_grouping(hamiltonian)
    if args.method == 'sorted_insertion':
        groups = baseline
    elif args.method in ('adhoc_repacking', 'adhoc_si_allocation'):
        groups = adhoc_repacking(hamiltonian, baseline)
    elif args.method == 'posthoc_repacking':
        groups = posthoc_repacking(hamiltonian, baseline)

    dmrg_energy = None
    mps = None
    exp_X = exp_Y = exp_Z = None

    if args.state_type == 'dmrg':
        ham_mpo = build_hamiltonian_mpo(hamiltonian, n_qubits, qubits, MPO_MAX_BOND)
        dmrg = qtn.DMRG2(ham_mpo, bond_dims=[DMRG_CHI])
        dmrg.solve(tol=1e-6, verbosity=0)
        mps = dmrg.state
        dmrg_energy = float(np.real(dmrg.energy))
        print(f"  DMRG energy: {dmrg_energy:.8f}")
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

    if args.method == 'sorted_insertion':
        shots = compute_optimal_allocation(si_sigmas, TOTAL_SHOTS)
        alloc_type = 'si_optimal'
    elif args.method in ('posthoc_repacking', 'adhoc_si_allocation'):
        shots = compute_optimal_allocation(si_sigmas, TOTAL_SHOTS)
        alloc_type = 'si_optimal'
    elif args.method == 'adhoc_repacking':
        warm = np.array(compute_optimal_allocation(si_sigmas, TOTAL_SHOTS))
        shots, _ = groups.shot_count_optimized(TOTAL_SHOTS, warm_start=warm)
        shots = list(shots)
        alloc_type = 'optimized'

    if args.state_type == 'dmrg':
        var = variance_mps(groups, n_qubits, qubits, mps, shots)
    else:
        var = variance_product_state(groups, exp_X, exp_Y, exp_Z, n_qubits, shots)

    print(f"  Variance: {var:.6e}")

    result = {
        'molecule': args.molecule, 'method': args.method, 'state_type': args.state_type,
        'allocation_type': alloc_type, 'n_qubits': n_qubits, 'n_electrons': n_electrons,
        'n_terms': hamiltonian.num_terms(), 'num_groups': groups.num_groups(),
        'total_variance': to_real(var), 'dmrg_energy': dmrg_energy,
    }
    fname = f"{args.molecule}_{args.state_type}_{args.method}_optimal_result.json"
    with open(OUTPUT_DIR / fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {fname}")


if __name__ == "__main__":
    main()
