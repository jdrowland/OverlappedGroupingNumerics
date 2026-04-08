class PauliString:
    __slots__ = ('x_bits', 'z_bits', 'coeff', 'n_qubits')

    def __init__(self, x_bits, z_bits, coeff=1.0, n_qubits=None):
        self.x_bits = x_bits
        self.z_bits = z_bits
        self.coeff = coeff
        if n_qubits is None:
            max_bit = max(x_bits, z_bits)
            self.n_qubits = max_bit.bit_length() if max_bit > 0 else 0
        else:
            self.n_qubits = n_qubits

    def copy(self):
        return PauliString(self.x_bits, self.z_bits, self.coeff, self.n_qubits)

    def __eq__(self, other):
        if not isinstance(other, PauliString):
            return False
        return (self.x_bits == other.x_bits and
                self.z_bits == other.z_bits and
                self.n_qubits == other.n_qubits)

    def __hash__(self):
        return hash((self.x_bits, self.z_bits, self.n_qubits))
