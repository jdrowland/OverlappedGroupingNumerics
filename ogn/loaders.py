import openfermion as of
from ogn.hamiltonian import Hamiltonian
from ogn.pauli import PauliString


def _from_openfermion(of_term, coeff, n_qubits):
    x_bits = 0
    z_bits = 0
    for qubit_idx, pauli in of_term:
        if pauli == 'X':
            x_bits |= (1 << qubit_idx)
        elif pauli == 'Y':
            x_bits |= (1 << qubit_idx)
            z_bits |= (1 << qubit_idx)
        elif pauli == 'Z':
            z_bits |= (1 << qubit_idx)
    return PauliString(x_bits, z_bits, coeff, n_qubits)


class HDF5Loader:
    def __init__(self, filepath, hermitianize=True):
        self.filepath = filepath
        self.hermitianize = hermitianize

    def load(self):
        mol_data = of.MolecularData(filename=self.filepath)
        mol_ham = mol_data.get_molecular_hamiltonian()
        fermion_ham = of.get_fermion_operator(mol_ham)
        qubit_ham = of.jordan_wigner(fermion_ham)
        n_qubits = of.count_qubits(qubit_ham)

        terms = [_from_openfermion(pauli_term, coeff, n_qubits)
                 for pauli_term, coeff in qubit_ham.terms.items()]

        metadata = {
            'source': 'HDF5',
            'filepath': self.filepath,
            'n_electrons': getattr(mol_data, 'n_electrons', None),
        }

        ham = Hamiltonian(terms, metadata)

        if self.hermitianize:
            ham, removed = ham.hermitianize()
            if removed:
                print(f"Hermitianization removed {len(removed)} term(s) from {self.filepath}")

        return ham
