import os
import numpy as np
from numba import njit, int64

from ogn.group import GroupCollection, PauliGroup
from ogn.sorted_insertion import _commutes, _build_block_masks

_H = 0
_S = 1
_CNOT = 2
_CZ = 3
_SWAP = 4


@njit(cache=True)
def _binary_gaussian_elimination_jit(matrix):
    n_rows, n_cols = matrix.shape
    mat = matrix.copy()
    next_row = 0

    for j in range(n_cols):
        found = False
        for i in range(next_row, n_rows):
            if mat[i, j]:
                found = True
                if i != next_row:
                    for c in range(n_cols):
                        mat[next_row, c], mat[i, c] = mat[i, c], mat[next_row, c]
                break

        if found:
            for i in range(next_row + 1, n_rows):
                if mat[i, j]:
                    for c in range(n_cols):
                        mat[i, c] ^= mat[next_row, c]
            next_row += 1

    return mat


@njit(cache=True)
def _get_independent_rows(stabilizer_matrix):
    n_paulis, n_cols = stabilizer_matrix.shape
    mat = np.zeros((n_cols, n_paulis), dtype=int64)
    for i in range(n_paulis):
        for j in range(n_cols):
            mat[j, i] = stabilizer_matrix[i, j]

    mat = _binary_gaussian_elimination_jit(mat)

    pivots = np.empty(min(n_paulis, n_cols), dtype=int64)
    n_pivots = 0
    next_pivot = 0
    for j in range(n_paulis):
        if next_pivot >= n_cols:
            break
        if mat[next_pivot, j]:
            pivots[n_pivots] = j
            n_pivots += 1
            next_pivot += 1

    return pivots[:n_pivots]


@njit(cache=True)
def _find_hadamard_qubits_jit(x_matrix, z_matrix, n_qubits, n_paulis):
    n_rows = 2 * n_qubits
    combined = np.zeros((n_rows, n_paulis), dtype=int64)
    for i in range(n_qubits):
        for j in range(n_paulis):
            combined[2 * i, j] = x_matrix[i, j]
            combined[2 * i + 1, j] = z_matrix[i, j]

    hadamard = np.zeros(n_qubits, dtype=int64)
    used = np.zeros(n_qubits, dtype=int64)

    for j in range(n_paulis):
        for row in range(n_rows):
            if combined[row, j] == 0:
                continue
            qubit = row // 2
            if used[qubit]:
                continue

            if row % 2 == 1:
                hadamard[qubit] = 1
            used[qubit] = 1

            for other in range(n_rows):
                if other != row and combined[other, j] == 1:
                    for p in range(n_paulis):
                        combined[other, p] = (combined[other, p] + combined[row, p]) % 2
            break

    return hadamard


@njit(cache=True)
def _diag_circuit_from_stabilizer(stabilizer_matrix, n_qubits):
    n_paulis = stabilizer_matrix.shape[0]

    x_matrix = np.zeros((n_qubits, n_paulis), dtype=int64)
    z_matrix = np.zeros((n_qubits, n_paulis), dtype=int64)
    for i in range(n_paulis):
        for q in range(n_qubits):
            x_matrix[q, i] = stabilizer_matrix[i, q]
            z_matrix[q, i] = stabilizer_matrix[i, q + n_qubits]

    max_ops = n_qubits * n_qubits + 4 * n_qubits
    ops = np.zeros((max_ops, 3), dtype=int64)
    n_ops = 0

    hadamard = _find_hadamard_qubits_jit(x_matrix, z_matrix, n_qubits, n_paulis)
    for q in range(n_qubits):
        if hadamard[q]:
            for p in range(n_paulis):
                x_matrix[q, p], z_matrix[q, p] = z_matrix[q, p], x_matrix[q, p]
            ops[n_ops, 0] = _H
            ops[n_ops, 1] = q
            ops[n_ops, 2] = 0
            n_ops += 1

    rank = min(n_paulis, n_qubits)

    for j in range(rank):
        if x_matrix[j, j] == 0:
            found_i = -1
            for i in range(j + 1, n_qubits):
                if x_matrix[i, j] != 0:
                    found_i = i
                    break
            if found_i >= 0:
                for p in range(n_paulis):
                    x_matrix[j, p], x_matrix[found_i, p] = x_matrix[found_i, p], x_matrix[j, p]
                    z_matrix[j, p], z_matrix[found_i, p] = z_matrix[found_i, p], z_matrix[j, p]
                ops[n_ops, 0] = _SWAP
                ops[n_ops, 1] = j
                ops[n_ops, 2] = found_i
                n_ops += 1

        for i in range(j + 1, n_qubits):
            if x_matrix[i, j] == 1:
                for p in range(n_paulis):
                    x_matrix[i, p] = (x_matrix[i, p] + x_matrix[j, p]) % 2
                    z_matrix[j, p] = (z_matrix[j, p] + z_matrix[i, p]) % 2
                ops[n_ops, 0] = _CNOT
                ops[n_ops, 1] = j
                ops[n_ops, 2] = i
                n_ops += 1

    for j in range(rank - 1, 0, -1):
        for i in range(j):
            if x_matrix[i, j] == 1:
                for p in range(n_paulis):
                    x_matrix[i, p] = (x_matrix[i, p] + x_matrix[j, p]) % 2
                    z_matrix[j, p] = (z_matrix[j, p] + z_matrix[i, p]) % 2
                ops[n_ops, 0] = _CNOT
                ops[n_ops, 1] = j
                ops[n_ops, 2] = i
                n_ops += 1

    for i in range(rank):
        if z_matrix[i, i] == 1:
            for p in range(n_paulis):
                z_matrix[i, p] = (z_matrix[i, p] + x_matrix[i, p]) % 2
            ops[n_ops, 0] = _S
            ops[n_ops, 1] = i
            ops[n_ops, 2] = 0
            n_ops += 1

        for j_cz in range(i):
            if z_matrix[i, j_cz] == 1:
                for p in range(n_paulis):
                    z_matrix[i, p] = (z_matrix[i, p] + x_matrix[j_cz, p]) % 2
                    z_matrix[j_cz, p] = (z_matrix[j_cz, p] + x_matrix[i, p]) % 2
                ops[n_ops, 0] = _CZ
                ops[n_ops, 1] = j_cz
                ops[n_ops, 2] = i
                n_ops += 1

    for i in range(rank):
        for p in range(n_paulis):
            x_matrix[i, p], z_matrix[i, p] = z_matrix[i, p], x_matrix[i, p]
        ops[n_ops, 0] = _H
        ops[n_ops, 1] = i
        ops[n_ops, 2] = 0
        n_ops += 1

    return ops[:n_ops], n_ops


def _diagonalize_group(paulis, n_qubits):
    n_paulis = len(paulis)
    stabilizer_matrix = np.zeros((n_paulis, 2 * n_qubits), dtype=np.int64)
    for i, p in enumerate(paulis):
        for q in range(n_qubits):
            if (p.x_bits >> q) & 1:
                stabilizer_matrix[i, q] = 1
            if (p.z_bits >> q) & 1:
                stabilizer_matrix[i, q + n_qubits] = 1
    pivot_indices = _get_independent_rows(stabilizer_matrix)
    return _diag_circuit_from_stabilizer(stabilizer_matrix[pivot_indices], n_qubits)


@njit(cache=True)
def _is_diagonal_jit(x, z, circuit_ops, n_ops):
    for i in range(n_ops):
        gate_type = circuit_ops[i, 0]
        q1 = circuit_ops[i, 1]
        q2 = circuit_ops[i, 2]

        if gate_type == 0:
            mask = int64(1) << q1
            xq = (x >> q1) & 1
            zq = (z >> q1) & 1
            x = (x & ~mask) | (zq << q1)
            z = (z & ~mask) | (xq << q1)
        elif gate_type == 1:
            xq = (x >> q1) & 1
            z ^= (xq << q1)
        elif gate_type == 2:
            xc = (x >> q1) & 1
            x ^= (xc << q2)
            zt = (z >> q2) & 1
            z ^= (zt << q1)
        elif gate_type == 3:
            xb = (x >> q2) & 1
            xa = (x >> q1) & 1
            z ^= (xb << q1)
            z ^= (xa << q2)
        elif gate_type == 4:
            mask_a = int64(1) << q1
            mask_b = int64(1) << q2
            xa = (x >> q1) & 1
            xb = (x >> q2) & 1
            x = (x & ~mask_a & ~mask_b) | (xb << q1) | (xa << q2)
            za = (z >> q1) & 1
            zb = (z >> q2) & 1
            z = (z & ~mask_a & ~mask_b) | (zb << q1) | (za << q2)

    return x == 0


@njit(cache=True)
def _posthoc_single_group(
    g, group_member_x, group_member_z, group_start, group_count,
    cand_x, cand_z, sort_order, group_offsets,
    circuit_ops_g, circuit_n_ops_g,
    block_masks, n_blocks
):
    start_g = group_start[g]
    count_g = group_count[g]
    n_to_check = group_offsets[g]
    results = np.empty(n_to_check, dtype=int64)
    n_found = int64(0)

    for ii in range(n_to_check):
        i = sort_order[ii]
        x = cand_x[i]
        z = cand_z[i]

        all_commute = True
        for m in range(count_g):
            x2 = group_member_x[start_g + m]
            z2 = group_member_z[start_g + m]
            if not _commutes(x, z, x2, z2, block_masks, n_blocks):
                all_commute = False
                break

        if not all_commute:
            continue

        if _is_diagonal_jit(x, z, circuit_ops_g, circuit_n_ops_g):
            results[n_found] = i
            n_found += 1

    return results[:n_found]


def posthoc_repacking(hamiltonian, baseline_groups, k=None):
    n_qubits = hamiltonian.num_qubits()
    n_groups = baseline_groups.num_groups()
    block_masks = _build_block_masks(n_qubits, k)

    all_ops = []
    for group in baseline_groups.groups:
        ops, n_ops = _diagonalize_group(group.paulis, n_qubits)
        all_ops.append((ops, n_ops))
    max_ops = max(o[1] for o in all_ops) if all_ops else 1
    circuit_ops = np.zeros((n_groups, max_ops, 3), dtype=np.int64)
    circuit_n_ops = np.zeros(n_groups, dtype=np.int64)
    for g, (ops, n_ops) in enumerate(all_ops):
        circuit_n_ops[g] = n_ops
        circuit_ops[g, :n_ops] = ops

    total_members = sum(len(g.paulis) for g in baseline_groups.groups)
    group_member_x = np.zeros(total_members, dtype=np.int64)
    group_member_z = np.zeros(total_members, dtype=np.int64)
    group_start = np.zeros(n_groups, dtype=np.int64)
    group_count = np.zeros(n_groups, dtype=np.int64)
    offset = 0
    for g_idx, group in enumerate(baseline_groups.groups):
        group_start[g_idx] = offset
        group_count[g_idx] = len(group.paulis)
        for p in group.paulis:
            group_member_x[offset] = p.x_bits
            group_member_z[offset] = p.z_bits
            offset += 1

    pauli_to_idx = {}
    all_paulis = []
    for p in hamiltonian.terms:
        key = (p.x_bits, p.z_bits)
        if key not in pauli_to_idx:
            pauli_to_idx[key] = len(all_paulis)
            all_paulis.append(p)

    n_candidates = len(all_paulis)
    cand_x = np.array([p.x_bits for p in all_paulis], dtype=np.int64)
    cand_z = np.array([p.z_bits for p in all_paulis], dtype=np.int64)

    cand_baseline_group = np.full(n_candidates, n_groups, dtype=np.int64)
    for g_idx, group in enumerate(baseline_groups.groups):
        for p in group.paulis:
            key = (p.x_bits, p.z_bits)
            if key in pauli_to_idx:
                idx = pauli_to_idx[key]
                if cand_baseline_group[idx] == n_groups:
                    cand_baseline_group[idx] = g_idx

    sort_order = np.argsort(cand_baseline_group).astype(np.int64)
    sorted_groups = cand_baseline_group[sort_order]
    group_offsets = np.searchsorted(sorted_groups, np.arange(n_groups, dtype=np.int64)).astype(np.int64)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_threads = max(1, int((os.cpu_count() or 1) * 0.7))
    per_group_added = [None] * n_groups
    per_group_added[0] = np.empty(0, dtype=np.int64)

    def process_group(g):
        if group_offsets[g] == 0:
            return g, np.empty(0, dtype=np.int64)
        return g, _posthoc_single_group(
            g, group_member_x, group_member_z, group_start, group_count,
            cand_x, cand_z, sort_order, group_offsets,
            circuit_ops[g], circuit_n_ops[g],
            block_masks, len(block_masks))

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_group, g) for g in range(1, n_groups)]
        for future in as_completed(futures):
            g, found = future.result()
            per_group_added[g] = found

    repacked = GroupCollection()
    for g_idx, group in enumerate(baseline_groups.groups):
        pg = PauliGroup()
        in_group = set()
        for p in group.paulis:
            pg.add(p.copy())
            in_group.add((p.x_bits, p.z_bits))
        if per_group_added[g_idx] is not None:
            for i in per_group_added[g_idx]:
                key = (int(cand_x[i]), int(cand_z[i]))
                if key not in in_group:
                    pg.add(all_paulis[i].copy())
                    in_group.add(key)
        repacked.groups.append(pg)

    return repacked
