import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import json
import time
from pathlib import Path

from ogn.pauli import PauliString
from ogn.hamiltonian import Hamiltonian
from ogn.sorted_insertion import sorted_insertion_grouping
from ogn.adhoc_repacking import adhoc_repacking
from ogn.posthoc_repacking import posthoc_repacking
from ogn.covariance_jit import compute_variance_with_covariance

SEED = 42
TOTAL_SHOTS = 100_000
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'results_scaling'
OUTPUT_DIR.mkdir(exist_ok=True)

def gen(n_qubits, seed=42):
    rng = np.random.RandomState(seed)
    total = 4**n_qubits - 1
    n_sample = int(np.ceil(total * 0.1))
    dim = 2**n_qubits
    indices = rng.choice(total, size=n_sample, replace=False) + 1
    coeffs = rng.uniform(-1.0, 1.0, size=n_sample)
    paulis = []
    for idx, c in zip(indices, coeffs):
        p = PauliString(int(idx % dim), int(idx // dim), float(c))
        p.n_qubits = n_qubits
        paulis.append(p)
    return paulis

def product_state_setup(n_qubits, seed=42):
    rng = np.random.RandomState(seed)
    thetas = rng.uniform(0, np.pi, size=n_qubits)
    phis = rng.uniform(0, 2 * np.pi, size=n_qubits)
    exp_X = np.sin(2 * thetas) * np.cos(phis)
    exp_Y = np.sin(2 * thetas) * np.sin(phis)
    exp_Z = np.cos(2 * thetas)
    return exp_X, exp_Y, exp_Z

def pauli_expectation(x, z, exp_X, exp_Y, exp_Z, n_qubits):
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
    return val

def compute_si_optimal_shots(groups, total_shots):
    sigmas = []
    for group in groups.groups:
        var_g = sum(float(np.real(p.coeff))**2
                    for p in group.paulis if not (p.x_bits == 0 and p.z_bits == 0))
        sigmas.append(np.sqrt(max(0.0, var_g)))
    sigmas = np.array(sigmas)
    s = sigmas.sum()
    if s == 0:
        return np.full(len(sigmas), total_shots / len(sigmas))
    return total_shots * sigmas / s

def compute_var_with_cov(groups, exp_X, exp_Y, exp_Z, n_qubits, shots):
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
            ge[offset] = pauli_expectation(p.x_bits, p.z_bits, exp_X, exp_Y, exp_Z, n_qubits)
            gNi[offset] = Ni_map[(p.x_bits, p.z_bits)]
            offset += 1
    return compute_variance_with_covariance(
        gx, gz, gc, ge, gNi, gs, gn, shots, exp_X, exp_Y, exp_Z, n_groups, n_qubits)

def main():
    for n in range(4, 13):
        print(f"\nn = {n}")
        t0 = time.time()
        paulis = gen(n)
        ham = Hamiltonian(paulis)
        exp_X, exp_Y, exp_Z = product_state_setup(n, seed=SEED)

        si_groups = sorted_insertion_grouping(ham)
        si_shots = compute_si_optimal_shots(si_groups, TOTAL_SHOTS)

        adhoc_groups = adhoc_repacking(ham, si_groups)
        posthoc_groups = posthoc_repacking(ham, si_groups)

        var_si = compute_var_with_cov(si_groups, exp_X, exp_Y, exp_Z, n, si_shots)
        var_adhoc_si = compute_var_with_cov(adhoc_groups, exp_X, exp_Y, exp_Z, n, si_shots)
        var_posthoc = compute_var_with_cov(posthoc_groups, exp_X, exp_Y, exp_Z, n, si_shots)

        var_adhoc_opt = None
        if len(paulis) <= 2_000_000:
            try:
                opt_shots, _ = adhoc_groups.shot_count_optimized(TOTAL_SHOTS)
                var_adhoc_opt = compute_var_with_cov(adhoc_groups, exp_X, exp_Y, exp_Z, n, opt_shots)
            except Exception as e:
                print(f"  Optimization failed: {e}")

        r_si = var_si / var_adhoc_si if var_adhoc_si > 0 else 0
        r_opt = var_si / var_adhoc_opt if var_adhoc_opt and var_adhoc_opt > 0 else 0
        r_ph = var_si / var_posthoc if var_posthoc > 0 else 0

        print(f"  {len(paulis)} terms, {len(si_groups.groups)} groups, {time.time()-t0:.1f}s")
        print(f"  adhoc_si={r_si:.4f}  adhoc_opt={r_opt:.4f}  posthoc={r_ph:.4f}")

        result = {
            'n_qubits': n, 'n_terms': len(paulis), 'num_groups': len(si_groups.groups),
            'var_si': var_si, 'var_adhoc_si': var_adhoc_si,
            'var_adhoc_opt': var_adhoc_opt, 'var_posthoc': var_posthoc,
            'ratio_adhoc_si': r_si, 'ratio_adhoc_opt': r_opt, 'ratio_posthoc': r_ph,
        }
        with open(OUTPUT_DIR / f'result_{n}q_all_methods_cov.json', 'w') as f:
            json.dump(result, f, indent=2)
        sys.stdout.flush()

if __name__ == '__main__':
    main()
