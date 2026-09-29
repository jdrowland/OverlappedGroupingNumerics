import sys
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ogn.covariance_jit import compute_variance_with_covariance

N_QUBITS = 44
N_ELECTRONS = 32
RANDOM_SEED = 42  # same product state convention as compute/run_optimal_allocation.py


def product_state(kind):
    if kind == 'hf':
        return np.zeros(N_QUBITS), np.zeros(N_QUBITS), np.array([-1.0 if i < N_ELECTRONS else 1.0 for i in range(N_QUBITS)])
    rng = np.random.RandomState(RANDOM_SEED)
    th, ph = rng.uniform(0, np.pi, N_QUBITS), rng.uniform(0, 2 * np.pi, N_QUBITS)
    return np.sin(2 * th) * np.cos(ph), np.sin(2 * th) * np.sin(ph), np.cos(2 * th)


def pauli_expectations(x, z, eX, eY, eZ):
    val = np.ones(len(x))
    for q in range(N_QUBITS):
        xq, zq = (x >> q) & 1, (z >> q) & 1
        val *= np.where(xq & zq, eY[q], np.where(xq, eX[q], np.where(zq, eZ[q], 1.0)))
    return val


def variance(arrs, shots, state):
    sizes = np.asarray(arrs['sizes'], np.int64)
    x, z, c = np.asarray(arrs['x'], np.int64), np.asarray(arrs['z'], np.int64), np.asarray(arrs['c'], np.float64)
    shots = np.asarray(shots, np.float64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    order = np.lexsort((z, x))
    first = np.ones(len(x), bool); first[1:] = (x[order][1:] != x[order][:-1]) | (z[order][1:] != z[order][:-1])
    uid = np.empty(len(x), np.int64); uid[order] = np.cumsum(first) - 1
    N = np.zeros(uid.max() + 1); np.add.at(N, uid, np.repeat(shots, sizes))
    eX, eY, eZ = state
    return compute_variance_with_covariance(x, z, c, pauli_expectations(x, z, eX, eY, eZ), N[uid],
                                            starts, sizes, shots, eX, eY, eZ, len(sizes), N_QUBITS)


def v1_arrays(pkl):
    d = pickle.load(open(pkl, 'rb'))
    sizes = np.array([len(v) for v in d['x_bits']], np.int64)
    return {'sizes': sizes, 'x': np.concatenate(d['x_bits']).astype(np.int64),
            'z': np.concatenate(d['z_bits']).astype(np.int64),
            'c': np.real(np.concatenate(d['coefficients'])).astype(np.float64)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('grouping')          # v1 .pkl (load_symplectic format) or new groupings_*.pkl + ':key'
    p.add_argument('shots_npy')
    p.add_argument('state', choices=['hf', 'random'])
    args = p.parse_args()
    t = time.time()
    if ':' in args.grouping:
        path, key = args.grouping.split(':')
        arrs = pickle.load(open(path, 'rb'))['groupings'][key]
    else:
        arrs = v1_arrays(args.grouping)
    v = variance(arrs, np.load(args.shots_npy), product_state(args.state))
    print(json.dumps({'grouping': args.grouping, 'shots': args.shots_npy, 'state': args.state,
                      'total_variance': float(v), 'seconds': round(time.time() - t, 1)}), flush=True)


if __name__ == '__main__':
    main()
