import numpy as np
from ogn.pauli import PauliString


class Hamiltonian:
    def __init__(self, terms, metadata=None):
        self.terms = list(terms)
        self.metadata = metadata or {}

    def num_terms(self):
        return len(self.terms)

    def num_qubits(self):
        if not self.terms:
            return 0
        return max(t.n_qubits for t in self.terms)

    def sort_by_coefficient(self, descending=True):
        sorted_terms = sorted(self.terms, key=lambda t: abs(t.coeff), reverse=descending)
        return Hamiltonian(sorted_terms, self.metadata)

    def hermitianize(self, prune_tol=1e-15):
        hermitian_terms = []
        removed_terms = []
        for pauli in self.terms:
            new_coeff = (pauli.coeff + np.conj(pauli.coeff)) / 2
            if abs(new_coeff) > prune_tol:
                if abs(np.imag(new_coeff)) < prune_tol:
                    new_coeff = np.real(new_coeff)
                hermitian_terms.append(PauliString(pauli.x_bits, pauli.z_bits, new_coeff, pauli.n_qubits))
            else:
                removed_terms.append((pauli, pauli.coeff))
        return Hamiltonian(hermitian_terms, self.metadata.copy()), removed_terms
