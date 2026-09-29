import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import openfermion as of

from ogn.loaders import _from_openfermion
from ogn.covariance_jit import _product_expectation

NX = 2
NY_VALUES = range(1, 11)
TUNNELING = 1.0
COULOMB = -1.0
MAX_STATES = 10_000
SEED = 42
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'product_state_varcov_v2.npz'


def hubbard_terms(ny):
    qubit_ham = of.jordan_wigner(of.fermi_hubbard(
        NX, ny, TUNNELING, COULOMB, periodic=True, spinless=True))
    n_qubits = of.count_qubits(qubit_ham)
    terms = [_from_openfermion(t, float(np.real(c)), n_qubits)
             for t, c in qubit_ham.terms.items() if t]
    return terms, n_qubits


def basis_states(n_qubits, rng):
    # All basis states when there are few enough, otherwise a sample drawn from
    # one RNG shared across system sizes (draw order ny=7,8,9,10).
    if 2**n_qubits <= MAX_STATES:
        return np.arange(2**n_qubits, dtype=np.int64)
    return rng.choice(2**n_qubits, MAX_STATES, replace=False).astype(np.int64)


def z_expectation(x_bits, z_bits, states):
    # <b|P|b> for a Pauli with no X/Y support; zero otherwise.
    if x_bits != 0:
        return np.zeros(len(states))
    parity = np.zeros(len(states), dtype=np.int64)
    masked = states & z_bits
    while np.any(masked):
        parity ^= masked & 1
        masked >>= 1
    return 1.0 - 2.0 * parity


def variance_covariance(terms, n_qubits, states):
    ones = np.ones(n_qubits)
    exps = [z_expectation(p.x_bits, p.z_bits, states) for p in terms]

    var = np.zeros(len(states))
    for p, e in zip(terms, exps):
        var += p.coeff**2 * (1.0 - e * e)

    cov = np.zeros(len(states))
    for i in range(len(terms)):
        pi = terms[i]
        for j in range(i + 1, len(terms)):
            pj = terms[j]
            anti = (pi.x_bits & pj.z_bits) ^ (pi.z_bits & pj.x_bits)
            if bin(anti).count('1') % 2:
                continue
            sign = _product_expectation(pi.x_bits, pi.z_bits, pj.x_bits, pj.z_bits,
                                        ones, ones, ones, n_qubits)
            prod = sign * z_expectation(pi.x_bits ^ pj.x_bits, pi.z_bits ^ pj.z_bits, states)
            cov += 2.0 * pi.coeff * pj.coeff * (prod - exps[i] * exps[j])

    return np.column_stack([var, cov])


def main():
    rng = np.random.RandomState(SEED)
    results = {}
    for ny in NY_VALUES:
        terms, n_qubits = hubbard_terms(ny)
        states = basis_states(n_qubits, rng)
        results[f'ny{ny}'] = variance_covariance(terms, n_qubits, states)
        d = results[f'ny{ny}']
        print(f"ny={ny}: {n_qubits}q {len(terms)} terms {len(states)} states  "
              f"var={d[:, 0].mean():.4f}  cov={d[:, 1].mean():+.4f}", flush=True)
    np.savez(OUTPUT_PATH, **results)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
