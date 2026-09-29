import numpy as np
from numba import njit, prange

# Same phase convention as variance_mps in compute/run_optimal_allocation.py
# (Pauli index p = 2*z + x; only the parity of the power matters for commuting pairs).
_PHASES = np.array([[0, 0, 0, 0], [0, 0, 1, 3], [0, 3, 0, 1], [0, 1, 3, 0]], dtype=np.int64)


def sorted_table(keys_x, keys_z, values):
    order = np.lexsort((keys_z, keys_x))
    return keys_x[order].astype(np.int64), keys_z[order].astype(np.int64), values[order].astype(np.float64)


@njit(cache=True)
def _find(kx, kz, x, z):
    lo, hi = 0, len(kx) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if kx[mid] < x or (kx[mid] == x and kz[mid] < z):
            lo = mid + 1
        elif kx[mid] == x and kz[mid] == z:
            return mid
        else:
            hi = mid - 1
    return -1


@njit(cache=True)
def _phase_sign(x1, z1, x2, z2, n_qubits):
    power = 0
    for q in range(n_qubits):
        p1 = 2 * ((z1 >> q) & 1) + ((x1 >> q) & 1)
        p2 = 2 * ((z2 >> q) & 1) + ((x2 >> q) & 1)
        power += _PHASES[p1, p2]
    return -1.0 if power % 4 == 2 else 1.0


@njit(cache=True)
def _lookup3(tables, x, z):
    if x == 0 and z == 0:
        return 1.0
    for t in range(len(tables)):
        kx, kz, kv = tables[t]
        i = _find(kx, kz, x, z)
        if i >= 0:
            return kv[i]
    raise KeyError("expectation value missing from all tables")


@njit(cache=True)
def _variance_multi(x, z, c, starts, sizes, shots, Ni, tables, n_qubits):
    total = 0.0
    for g in range(len(sizes)):
        ng = shots[g]
        if ng <= 0:
            continue
        s, m = starts[g], sizes[g]
        e = np.empty(m)
        for a in range(m):
            e[a] = _lookup3(tables, x[s + a], z[s + a])
            total += c[s + a] * c[s + a] * ng / (Ni[s + a] * Ni[s + a]) * (1.0 - e[a] * e[a])
        for a in range(m):
            for b in range(a + 1, m):
                px, pz = x[s + a] ^ x[s + b], z[s + a] ^ z[s + b]
                ek = _phase_sign(x[s + a], z[s + a], x[s + b], z[s + b], n_qubits) * _lookup3(tables, px, pz)
                total += 2.0 * c[s + a] * c[s + b] * ng / (Ni[s + a] * Ni[s + b]) * (ek - e[a] * e[b])
    return total


def variance_lookup_multi(arrs, shots, tables, n_qubits, parallel=False):
    """arrs: {'sizes','x','z','c'} flattened grouping; tables: sorted, disjoint (kx, kz, kv) tables of <P>."""
    from numba.typed import List
    sizes = np.asarray(arrs['sizes'], np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    shots = np.asarray(shots, np.float64)
    ux, uz, _ = sorted_table(arrs['x'], arrs['z'], np.zeros(len(arrs['x'])))
    keep = np.ones(len(ux), bool); keep[1:] = (ux[1:] != ux[:-1]) | (uz[1:] != uz[:-1])
    ux, uz = ux[keep], uz[keep]
    idx = np.array([_find(ux, uz, xx, zz) for xx, zz in zip(arrs['x'], arrs['z'])])
    N = np.zeros(len(ux)); np.add.at(N, idx, np.repeat(shots, sizes))
    tl = List()
    for t in tables:
        tl.append((np.asarray(t[0], np.int64), np.asarray(t[1], np.int64), np.asarray(t[2], np.float64)))
    kernel = _variance_multi_parallel if parallel else _variance_multi
    v = kernel(np.asarray(arrs['x'], np.int64), np.asarray(arrs['z'], np.int64),
               np.asarray(arrs['c'], np.float64), starts, sizes, shots, N[idx], tl, n_qubits)
    if np.isnan(v):
        raise KeyError("expectation value missing from all tables")
    return v


@njit(cache=True)
def _lookup_or_nan(tables, x, z):
    # exceptions raised inside prange do not propagate, so a missing key becomes NaN and is checked by the caller
    if x == 0 and z == 0:
        return 1.0
    for t in range(len(tables)):
        kx, kz, kv = tables[t]
        i = _find(kx, kz, x, z)
        if i >= 0:
            return kv[i]
    return np.nan


@njit(cache=True, parallel=True)
def _variance_multi_parallel(x, z, c, starts, sizes, shots, Ni, tables, n_qubits):
    part = np.zeros(len(sizes))
    for g in prange(len(sizes)):
        ng = shots[g]
        if ng <= 0:
            continue
        s, m = starts[g], sizes[g]
        e = np.empty(m)
        acc = 0.0
        for a in range(m):
            e[a] = _lookup_or_nan(tables, x[s + a], z[s + a])
            acc += c[s + a] * c[s + a] * ng / (Ni[s + a] * Ni[s + a]) * (1.0 - e[a] * e[a])
        for a in range(m):
            for b in range(a + 1, m):
                px, pz = x[s + a] ^ x[s + b], z[s + a] ^ z[s + b]
                ek = _phase_sign(x[s + a], z[s + a], x[s + b], z[s + b], n_qubits) * _lookup_or_nan(tables, px, pz)
                acc += 2.0 * c[s + a] * c[s + b] * ng / (Ni[s + a] * Ni[s + b]) * (ek - e[a] * e[b])
        part[g] = acc
    return part.sum()
