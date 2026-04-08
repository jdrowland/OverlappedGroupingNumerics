import heapq
import numpy as np
from numba import njit, int64

from ogn.hamiltonian import Hamiltonian
from ogn.group import GroupCollection, PauliGroup
from ogn.sorted_insertion import _commutes, _build_block_masks


@njit(cache=True)
def _find_compatible_group(
    x, z,
    group_member_x, group_member_z, group_count,
    group_gx, group_gz, group_valid,
    start_group, n_groups, block_masks, n_blocks
):
    support = x | z
    for g in range(start_group, n_groups):
        gx = group_gx[g]
        gz = group_gz[g]
        valid = group_valid[g]

        anticommute = (x & gz) ^ (z & gx)
        invalid_support = support & ~valid

        if invalid_support == 0 and (anticommute & valid) == 0:
            return g

        count = group_count[g]
        all_commute = True
        for m in range(count):
            x2 = group_member_x[g, m]
            z2 = group_member_z[g, m]
            if not _commutes(x, z, x2, z2, block_masks, n_blocks):
                all_commute = False
                break

        if all_commute:
            return g

    return int64(-1)


def adhoc_repacking(hamiltonian, baseline_groups, k=None):
    n_qubits = hamiltonian.num_qubits()
    n_groups = len(baseline_groups.groups)
    all_bits_valid = (1 << n_qubits) - 1
    block_masks = _build_block_masks(n_qubits, k)
    n_blocks = len(block_masks)

    max_group_size = len(hamiltonian.terms)
    group_member_x = np.zeros((n_groups, max_group_size), dtype=np.int64)
    group_member_z = np.zeros((n_groups, max_group_size), dtype=np.int64)
    group_count = np.zeros(n_groups, dtype=np.int64)
    group_gx = np.zeros(n_groups, dtype=np.int64)
    group_gz = np.zeros(n_groups, dtype=np.int64)
    group_valid = np.zeros(n_groups, dtype=np.int64)

    for g_idx, group in enumerate(baseline_groups.groups):
        gx = 0
        gz = 0
        valid = all_bits_valid
        for i, p in enumerate(group.paulis):
            x, z = p.x_bits, p.z_bits
            group_member_x[g_idx, i] = x
            group_member_z[g_idx, i] = z
            anticommute = (x & gz) ^ (z & gx)
            gx |= x
            gz |= z
            valid &= ~anticommute
        group_count[g_idx] = len(group.paulis)
        group_gx[g_idx] = gx
        group_gz[g_idx] = gz
        group_valid[g_idx] = valid

    term_key_to_idx = {}
    for t_idx, t in enumerate(hamiltonian.terms):
        term_key_to_idx[(t.x_bits, t.z_bits)] = t_idx

    pauli_groups = [[] for _ in range(len(hamiltonian.terms))]
    for g_idx, group in enumerate(baseline_groups.groups):
        for p in group.paulis:
            key = (p.x_bits, p.z_bits)
            if key in term_key_to_idx:
                pauli_groups[term_key_to_idx[key]].append(g_idx)

    pq = []
    for t_idx, term in enumerate(hamiltonian.terms):
        n_measurements = len(pauli_groups[t_idx]) if pauli_groups[t_idx] else 1
        priority = abs(term.coeff) ** 2 / n_measurements
        heapq.heappush(pq, (-priority, t_idx))

    while pq:
        neg_priority, t_idx = heapq.heappop(pq)
        term = hamiltonian.terms[t_idx]

        current_groups = pauli_groups[t_idx]
        if not current_groups:
            continue
        max_group_idx = max(current_groups)

        g = _find_compatible_group(
            np.int64(term.x_bits), np.int64(term.z_bits),
            group_member_x, group_member_z, group_count,
            group_gx, group_gz, group_valid,
            max_group_idx + 1, n_groups, block_masks, n_blocks
        )

        if g >= 0:
            count = group_count[g]
            x, z = term.x_bits, term.z_bits
            group_member_x[g, count] = x
            group_member_z[g, count] = z

            anticommute = (x & group_gz[g]) ^ (z & group_gx[g])
            group_gx[g] |= x
            group_gz[g] |= z
            group_valid[g] &= ~anticommute
            group_count[g] = count + 1

            pauli_groups[t_idx].append(g)

            n_measurements = len(pauli_groups[t_idx])
            priority = abs(term.coeff) ** 2 / n_measurements
            heapq.heappush(pq, (-priority, t_idx))

    repacked = GroupCollection()
    for g_idx, group in enumerate(baseline_groups.groups):
        pg = PauliGroup()
        for p in group.paulis:
            pg.add(p.copy())
        orig_count = len(group.paulis)
        for m in range(orig_count, group_count[g_idx]):
            x = int(group_member_x[g_idx, m])
            z = int(group_member_z[g_idx, m])
            key = (x, z)
            if key in term_key_to_idx:
                t_idx = term_key_to_idx[key]
                pg.add(hamiltonian.terms[t_idx].copy())
        repacked.groups.append(pg)

    return repacked
