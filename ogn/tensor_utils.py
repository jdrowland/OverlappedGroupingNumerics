import numpy as np
import cirq
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct


def pauli_string_to_mpo(pstring, qs):
    ps_dense = pstring.dense(qs)
    tensors = []
    n = len(ps_dense.pauli_mask)
    for i, pauli_int in enumerate(ps_dense.pauli_mask):
        if pauli_int == 0:
            m = np.eye(2)
        elif pauli_int == 1:
            m = cirq.unitary(cirq.X)
        elif pauli_int == 2:
            m = cirq.unitary(cirq.Y)
        else:
            m = cirq.unitary(cirq.Z)
        if i == 0:
            tensors.append(m.reshape((2, 2, 1)))
        elif i == n - 1:
            tensors.append(m.reshape((1, 2, 2)))
        else:
            tensors.append(m.reshape((1, 2, 2, 1)))
    return pstring.coefficient * MatrixProductOperator(tensors, shape="ludr")


def pauli_sum_to_mpo(psum, qs, max_bond):
    for i, p in enumerate(psum):
        if i == 0:
            mpo = pauli_string_to_mpo(p, qs)
        else:
            mpo += pauli_string_to_mpo(p, qs)
            tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)
    return mpo


def mpo_mps_expectation(mpo, mps):
    return mps.H @ mpo.apply(mps.copy())
