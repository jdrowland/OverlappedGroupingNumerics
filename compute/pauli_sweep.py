import numpy as np

def mps_site_arrays(mps):
    """Site tensors of a quimb MPS reshaped to (left bond, right bond, physical)."""
    out = []
    n = mps.L
    for i in range(n):
        t = mps[i]
        inds = [mps.bond(i - 1, i)] if i > 0 else []
        inds += [mps.bond(i, i + 1)] if i < n - 1 else []
        a = np.asarray(t.transpose(*inds, mps.site_ind(i)).data)
        if i == 0:
            a = a[None, :, :]
        if i == n - 1:
            a = a[:, None, :]
        out.append(np.ascontiguousarray(a))
    return out


def batch_expectations(sites, x_bits, z_bits, batch=1000):
    """<psi|P|psi> for many Pauli strings at once: environments stacked as (B, chi, chi)."""
    x_bits, z_bits = np.asarray(x_bits, np.int64), np.asarray(z_bits, np.int64)
    out = np.empty(len(x_bits))
    for s0 in range(0, len(x_bits), batch):
        xb, zb = x_bits[s0:s0 + batch], z_bits[s0:s0 + batch]
        B = len(xb)
        E = np.ones((B, 1, 1), dtype=complex)
        for q, A in enumerate(sites):
            cl, cr = A.shape[0], A.shape[1]
            T = np.matmul(E, A.reshape(cl, cr * 2)).reshape(B, cl, cr, 2)          # (B, l_bra, r_ket, s')
            xq, zq = (xb >> q) & 1, (zb >> q) & 1
            T2 = T.copy()
            m = (xq == 1) & (zq == 0)
            if m.any():
                T2[m] = T[m][..., ::-1]
            m = (xq == 0) & (zq == 1)
            if m.any():
                T2[m, ..., 1] *= -1
            m = (xq == 1) & (zq == 1)
            if m.any():
                T2[m, ..., 0] = -1j * T[m, ..., 1]
                T2[m, ..., 1] = 1j * T[m, ..., 0]
            Ac = A.conj().transpose(1, 0, 2).reshape(cr, cl * 2)                     # (r_bra, l_bra*s)
            E = np.matmul(Ac, T2.transpose(0, 1, 3, 2).reshape(B, cl * 2, cr))       # (B, r_bra, r_ket)
        out[s0:s0 + B] = np.real(E[:, 0, 0])
    return out

