import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent.parent))

from ogn.pauli import PauliString
from ogn.hamiltonian import Hamiltonian
from ogn.sorted_insertion import sorted_insertion_grouping
from ogn.adhoc_repacking import adhoc_repacking
from ogn.posthoc_repacking import posthoc_repacking

N_QUBITS = 44
TIE_DECIMALS = 10  # must match compute/run_optimal_allocation.py
BATCH_PAIRS = 50_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_owp(path):
    d = np.load(path)
    terms = [PauliString(int(x), int(z), float(c), N_QUBITS)
             for x, z, c in zip(d['x_bits'], d['z_bits'], d['coeffs'])]
    return Hamiltonian(terms)


def permute_ties(hamiltonian, rng):
    terms = hamiltonian.terms
    keys = rng.random(len(terms))
    order = sorted(range(len(terms)), key=lambda i: (-round(abs(terms[i].coeff), TIE_DECIMALS), keys[i]))
    return Hamiltonian([terms[i] for i in order], hamiltonian.metadata)


def to_arrays(groups):
    sizes = np.array([len(g.paulis) for g in groups.groups], dtype=np.int64)
    x = np.array([p.x_bits for g in groups.groups for p in g.paulis], dtype=np.int64)
    z = np.array([p.z_bits for g in groups.groups for p in g.paulis], dtype=np.int64)
    c = np.array([p.coeff for g in groups.groups for p in g.paulis], dtype=np.float64)
    return {'sizes': sizes, 'x': x, 'z': z, 'c': c}


@njit(cache=True)
def _pair_products(x, z, starts, sizes, g0, g1, out_x, out_z):
    n = 0
    for g in range(g0, g1):
        s, m = starts[g], sizes[g]
        for i in range(m):
            for j in range(i + 1, m):
                out_x[n] = x[s + i] ^ x[s + j]
                out_z[n] = z[s + i] ^ z[s + j]
                n += 1
    return n


def unique_keys(kx, kz):
    order = np.lexsort((kz, kx))
    kx, kz = kx[order], kz[order]
    keep = np.ones(len(kx), dtype=bool)
    keep[1:] = (kx[1:] != kx[:-1]) | (kz[1:] != kz[:-1])
    return kx[keep], kz[keep]


def enumerate_products(arrs_list):
    ux, uz = [], []
    for arrs in arrs_list:
        sizes = arrs['sizes']; starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        pairs = sizes * (sizes - 1) // 2
        g = 0
        while g < len(sizes):
            g1, tot = g, 0
            while g1 < len(sizes) and (tot == 0 or tot + pairs[g1] <= BATCH_PAIRS):
                tot += pairs[g1]; g1 += 1
            ox, oz = np.empty(tot, np.int64), np.empty(tot, np.int64)
            n = _pair_products(arrs['x'], arrs['z'], starts, sizes, g, g1, ox, oz)
            bx, bz = unique_keys(ox[:n], oz[:n])
            ux.append(bx); uz.append(bz)
            g = g1
        bx, bz = unique_keys(np.concatenate(ux), np.concatenate(uz)); ux, uz = [bx], [bz]
    return ux[0], uz[0], int(sum(int((a['sizes'] * (a['sizes'] - 1) // 2).sum()) for a in arrs_list))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('owp_npz')
    p.add_argument('out_dir')
    p.add_argument('sample', type=int)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ham = load_owp(args.owp_npz)
    log(f"loaded {ham.num_terms()} terms, {ham.num_qubits()} qubits")
    ham = permute_ties(ham, np.random.default_rng([args.seed, args.sample]))
    base = sorted_insertion_grouping(ham, sort_descending=False)
    log(f"sorted insertion: {base.num_groups()} groups ({time.time()-t0:.0f}s)")
    t = time.time(); adhoc = adhoc_repacking(ham, base)
    log(f"ad-hoc: {sum(len(g.paulis) for g in adhoc.groups)} members ({time.time()-t:.0f}s)")
    t = time.time(); post = posthoc_repacking(ham, base)
    log(f"post-hoc: {sum(len(g.paulis) for g in post.groups)} members ({time.time()-t:.0f}s)")

    arrs = {name: to_arrays(g) for name, g in [('sorted_insertion', base), ('adhoc', adhoc), ('posthoc', post)]}
    with open(out / f'groupings_seed{args.seed}_sample{args.sample}.pkl', 'wb') as f:
        pickle.dump({'seed': [args.seed, args.sample], 'tie_decimals': TIE_DECIMALS, 'groupings': arrs}, f)

    # SI groups are subsets of the ad-hoc groups, so ad-hoc and post-hoc pairs cover all products.
    t = time.time()
    kx, kz, n_pairs = enumerate_products([arrs['adhoc'], arrs['posthoc']])
    np.savez(out / f'products_seed{args.seed}_sample{args.sample}.npz', x=kx, z=kz)
    log(f"products: {n_pairs} pairs -> {len(kx)} unique ({time.time()-t:.0f}s); total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
