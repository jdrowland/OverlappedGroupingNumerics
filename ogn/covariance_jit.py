import numpy as np
from numba import njit, int64

_PHASE_TABLE = np.array([
    [0, 0, 0, 0],
    [0, 0, 3, 1],
    [0, 1, 0, 3],
    [0, 3, 1, 0],
], dtype=np.int64)


@njit(cache=True)
def _product_expectation(x1, z1, x2, z2, exp_X, exp_Y, exp_Z, n_qubits):
    phase_power = int64(0)
    exp_val = 1.0

    for q in range(n_qubits):
        xq1 = (x1 >> q) & 1
        zq1 = (z1 >> q) & 1
        xq2 = (x2 >> q) & 1
        zq2 = (z2 >> q) & 1

        p1 = int64(2 * zq1 + xq1)
        p2 = int64(2 * zq2 + xq2)

        phase_power += _PHASE_TABLE[p1, p2]

        xq_prod = xq1 ^ xq2
        zq_prod = zq1 ^ zq2

        if xq_prod and zq_prod:
            exp_val *= exp_Y[q]
        elif xq_prod:
            exp_val *= exp_X[q]
        elif zq_prod:
            exp_val *= exp_Z[q]

    if phase_power % 4 == 2:
        return -exp_val
    return exp_val


@njit(cache=True)
def compute_variance_with_covariance(
    group_x_flat, group_z_flat, group_coeffs_flat,
    group_exp_flat, group_Ni_flat,
    group_start, group_count,
    shots_per_group,
    exp_X, exp_Y, exp_Z,
    n_groups, n_qubits
):
    total_var = 0.0

    for g in range(n_groups):
        start = group_start[g]
        count = group_count[g]
        ng = shots_per_group[g]
        if ng <= 0:
            continue

        group_var = 0.0

        for i in range(count):
            idx = start + i
            c = group_coeffs_flat[idx]
            e = group_exp_flat[idx]
            Ni = group_Ni_flat[idx]
            if Ni > 0:
                group_var += c * c * ng / (Ni * Ni) * (1.0 - e * e)

        for i in range(count):
            for j in range(i + 1, count):
                idx_i = start + i
                idx_j = start + j
                Ni = group_Ni_flat[idx_i]
                Nj = group_Ni_flat[idx_j]
                if Ni <= 0 or Nj <= 0:
                    continue
                exp_prod = _product_expectation(
                    group_x_flat[idx_i], group_z_flat[idx_i],
                    group_x_flat[idx_j], group_z_flat[idx_j],
                    exp_X, exp_Y, exp_Z, n_qubits
                )
                cov = exp_prod - group_exp_flat[idx_i] * group_exp_flat[idx_j]
                group_var += 2.0 * group_coeffs_flat[idx_i] * group_coeffs_flat[idx_j] * ng / (Ni * Nj) * cov

        total_var += group_var

    return total_var
