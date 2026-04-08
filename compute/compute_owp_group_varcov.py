import sys
import time
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openfermion as of
from ogn.group import GroupCollection
from ogn.tensor_utils import pauli_sum_to_mpo, mpo_mps_expectation
import cirq

BASE = Path(__file__).parent.parent
OWP_DIR = BASE / 'data' / 'owp'
N_QUBITS = 44
MAX_BOND = 64

_PHASES = [[0,0,0,0],[0,0,1,3],[0,3,0,1],[0,1,3,0]]


def load_mps(chi):
    with open(OWP_DIR / f'owp_reactant_chi{chi}.pkl', 'rb') as f:
        return pickle.load(f)


def compute_exp(x_bits, z_bits, qubits, mps, cache):
    key = (x_bits, z_bits)
    if key in cache:
        return cache[key]
    if x_bits == 0 and z_bits == 0:
        cache[key] = 1.0
        return 1.0
    term = []
    for i in range(N_QUBITS):
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
    mpo = pauli_sum_to_mpo(psum, qubits, MAX_BOND)
    val = float(np.real(mpo_mps_expectation(mpo, mps)))
    cache[key] = val
    return val


def main():
    if len(sys.argv) < 4:
        print("Usage: python compute_owp_group_varcov.py <grouping_type> <chi> <group_indices>")
        print("  Allocation .npy files are loaded from data/owp/")
        print("  group_indices: comma-separated")
        sys.exit(1)

    grouping_type = sys.argv[1]
    chi = int(sys.argv[2])
    group_indices = [int(x) for x in sys.argv[3].split(',')]

    print(f"OWP varcov: {grouping_type} chi={chi} groups={group_indices}")

    gc = GroupCollection.load_symplectic(str(OWP_DIR / f'groups_{grouping_type}.pkl'))

    alloc_files = [OWP_DIR / 'si_shots.npy']
    if grouping_type == 'repacked':
        alloc_files.append(OWP_DIR / 'opt_shots.npy')

    allocs = []
    for af in alloc_files:
        shots = np.load(af)
        Ni_map = {}
        for g_idx, group in enumerate(gc.groups):
            for p in group.paulis:
                key = (p.x_bits, p.z_bits)
                Ni_map[key] = Ni_map.get(key, 0.0) + shots[g_idx]
        allocs.append((af.stem, list(shots), Ni_map))

    mps = load_mps(chi)
    qubits = cirq.LineQubit.range(N_QUBITS)
    exp_cache = {}

    out_dir = OWP_DIR / f'results_{grouping_type}' / f'chi{chi}_weighted'
    out_dir.mkdir(parents=True, exist_ok=True)

    for group_idx in group_indices:
        out_file = out_dir / f'group_{group_idx}.pkl'
        if out_file.exists():
            print(f"  group {group_idx}: exists, skipping")
            continue

        group = gc.groups[group_idx]
        paulis = group.paulis
        n_terms = len(paulis)
        t0 = time.time()

        per_alloc_diag = [0.0] * len(allocs)
        per_alloc_cov = [0.0] * len(allocs)

        for p in paulis:
            exp_val = compute_exp(p.x_bits, p.z_bits, qubits, mps, exp_cache)
            var_i = max(0.0, 1.0 - exp_val ** 2)
            c_i = float(np.real(p.coeff))
            for a_idx, (name, shots, Ni_map) in enumerate(allocs):
                ng = shots[group_idx]
                Ni = Ni_map[(p.x_bits, p.z_bits)]
                if Ni > 0 and ng > 0:
                    per_alloc_diag[a_idx] += c_i * c_i * ng / (Ni * Ni) * var_i

        for i in range(n_terms):
            pi = paulis[i]
            ci = float(np.real(pi.coeff))
            ei = exp_cache[(pi.x_bits, pi.z_bits)]
            for j in range(i + 1, n_terms):
                pj = paulis[j]
                cj = float(np.real(pj.coeff))
                ej = exp_cache[(pj.x_bits, pj.z_bits)]

                prod_x = pi.x_bits ^ pj.x_bits
                prod_z = pi.z_bits ^ pj.z_bits
                power = 0
                for q in range(N_QUBITS):
                    p1 = 2 * ((pi.z_bits >> q) & 1) + ((pi.x_bits >> q) & 1)
                    p2 = 2 * ((pj.z_bits >> q) & 1) + ((pj.x_bits >> q) & 1)
                    power += _PHASES[p1][p2]
                phase_sign = -1.0 if power % 4 == 2 else 1.0

                ek = compute_exp(prod_x, prod_z, qubits, mps, exp_cache)
                cov = phase_sign * ek - ei * ej

                for a_idx, (name, shots, Ni_map) in enumerate(allocs):
                    ng = shots[group_idx]
                    Ni = Ni_map[(pi.x_bits, pi.z_bits)]
                    Nj = Ni_map[(pj.x_bits, pj.z_bits)]
                    if Ni > 0 and Nj > 0 and ng > 0:
                        per_alloc_cov[a_idx] += 2.0 * ci * cj * ng / (Ni * Nj) * cov

        elapsed = time.time() - t0

        result = {
            'group_idx': group_idx,
            'grouping_type': grouping_type,
            'chi': chi,
            'n_terms': n_terms,
            'n_pairs': n_terms * (n_terms - 1) // 2,
            'total_time': elapsed,
            'allocations': {},
        }
        for a_idx, (name, shots, Ni_map) in enumerate(allocs):
            result['allocations'][name] = {
                'diag': per_alloc_diag[a_idx],
                'cov': per_alloc_cov[a_idx],
                'full': per_alloc_diag[a_idx] + per_alloc_cov[a_idx],
            }

        with open(out_file, 'wb') as f:
            pickle.dump(result, f)

        alloc_str = ' '.join(f'{name}={per_alloc_diag[a]+per_alloc_cov[a]:.6e}' for a, (name,_,_) in enumerate(allocs))
        print(f"  group {group_idx}: {n_terms} terms, {elapsed:.1f}s, {alloc_str}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
