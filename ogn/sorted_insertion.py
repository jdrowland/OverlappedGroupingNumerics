import numpy as np
from numba import njit, int64

from ogn.hamiltonian import Hamiltonian
from ogn.group import GroupCollection, PauliGroup


@njit(cache=True)
def _popcount(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


@njit(cache=True)
def _commutes(x, z, x2, z2, block_masks, n_blocks):
    anti = (x & z2) ^ (z & x2)
    if anti == 0:
        return True
    for b in range(n_blocks):
        if _popcount(anti & block_masks[b]) & 1:
            return False
    return True


@njit(cache=True)
def _si_core(x_arr, z_arr, n_terms, n_qubits, block_masks, n_blocks):
    max_groups = n_terms
    max_group_size = min(n_terms, 4096)

    group_gx = np.zeros(max_groups, dtype=int64)
    group_gz = np.zeros(max_groups, dtype=int64)
    group_valid = np.zeros(max_groups, dtype=int64)
    group_member_x = np.zeros((max_groups, max_group_size), dtype=int64)
    group_member_z = np.zeros((max_groups, max_group_size), dtype=int64)
    group_count = np.zeros(max_groups, dtype=int64)

    all_bits_valid = int64((1 << n_qubits) - 1)
    assignments = np.full(n_terms, -1, dtype=int64)
    n_groups = 0

    for i in range(n_terms):
        x = x_arr[i]
        z = z_arr[i]
        support = x | z
        placed = False

        for g in range(n_groups):
            gx = group_gx[g]
            gz = group_gz[g]
            valid = group_valid[g]

            anticommute = (x & gz) ^ (z & gx)
            invalid_support = support & ~valid

            if invalid_support == 0 and (anticommute & valid) == 0:
                count = group_count[g]
                assignments[i] = g
                group_gx[g] = gx | x
                group_gz[g] = gz | z
                group_member_x[g, count] = x
                group_member_z[g, count] = z
                group_count[g] = count + 1
                placed = True
                break

            count = group_count[g]
            all_commute = True
            for m in range(count):
                x2 = group_member_x[g, m]
                z2 = group_member_z[g, m]
                if not _commutes(x, z, x2, z2, block_masks, n_blocks):
                    all_commute = False
                    break

            if all_commute:
                assignments[i] = g
                group_gx[g] = gx | x
                group_gz[g] = gz | z
                group_valid[g] = valid & ~anticommute
                group_member_x[g, count] = x
                group_member_z[g, count] = z
                group_count[g] = count + 1
                placed = True
                break

        if not placed:
            g = n_groups
            group_gx[g] = x
            group_gz[g] = z
            group_valid[g] = all_bits_valid
            group_member_x[g, 0] = x
            group_member_z[g, 0] = z
            group_count[g] = 1
            assignments[i] = g
            n_groups += 1

    return assignments, n_groups


def _build_block_masks(n_qubits, k):
    if k is None:
        return np.array([(1 << n_qubits) - 1], dtype=np.int64)
    n_blocks = (n_qubits + k - 1) // k
    masks = np.zeros(n_blocks, dtype=np.int64)
    for block_idx in range(n_blocks):
        start = block_idx * k
        end = min(start + k, n_qubits)
        masks[block_idx] = ((1 << end) - 1) ^ ((1 << start) - 1)
    return masks


def sorted_insertion_grouping(hamiltonian, k=None, sort_descending=True):
    if sort_descending:
        sorted_ham = hamiltonian.sort_by_coefficient(descending=True)
        terms = sorted_ham.terms
    else:
        terms = hamiltonian.terms

    n_terms = len(terms)
    n_qubits = hamiltonian.num_qubits()

    x_arr = np.array([t.x_bits for t in terms], dtype=np.int64)
    z_arr = np.array([t.z_bits for t in terms], dtype=np.int64)
    block_masks = _build_block_masks(n_qubits, k)

    assignments, n_groups = _si_core(x_arr, z_arr, n_terms, n_qubits,
                                     block_masks, len(block_masks))

    groups = GroupCollection()
    group_lists = [[] for _ in range(n_groups)]
    for i, term in enumerate(terms):
        group_lists[assignments[i]].append(term)

    for gl in group_lists:
        group = PauliGroup()
        for pauli in gl:
            group.add(pauli)
        groups.groups.append(group)

    return groups
