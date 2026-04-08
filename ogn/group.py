import numpy as np


class PauliGroup:
    def __init__(self):
        self.paulis = []
        self.gx = 0
        self.gz = 0
        self.valid = 0

    def add(self, pauli):
        self.paulis.append(pauli)
        self.gx |= pauli.x_bits
        self.gz |= pauli.z_bits


class GroupCollection:
    def __init__(self):
        self.groups = []

    def num_groups(self):
        return len(self.groups)

    def shot_count_optimized(self, total_shots, warm_start=None, verbose=False):
        from scipy.optimize import minimize
        from scipy.sparse import csr_matrix

        num_groups = self.num_groups()

        pauli_groups = {}
        pauli_csq = {}
        for g_idx, group in enumerate(self.groups):
            for p in group.paulis:
                key = (p.x_bits, p.z_bits)
                if key not in pauli_groups:
                    pauli_groups[key] = set()
                    pauli_csq[key] = abs(p.coeff) ** 2
                pauli_groups[key].add(g_idx)

        pattern_data = {}
        for key in pauli_groups:
            pat = frozenset(pauli_groups[key])
            pattern_data[pat] = pattern_data.get(pat, 0.0) + pauli_csq[key]

        num_patterns = len(pattern_data)
        rows, cols = [], []
        weights = np.zeros(num_patterns)
        for i, (pat, csq_sum) in enumerate(pattern_data.items()):
            weights[i] = csq_sum
            for g in pat:
                rows.append(i)
                cols.append(g)

        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_patterns, num_groups))
        A_T = A.T.tocsr()

        def softmax(theta):
            e = np.exp(theta - theta.max())
            return e / e.sum()

        def objective_theta(theta):
            n = total_shots * softmax(theta)
            t = np.maximum(A.dot(n), 1e-12)
            return np.sum(weights / t)

        def gradient_theta(theta):
            s = softmax(theta)
            n = total_shots * s
            t = np.maximum(A.dot(n), 1e-12)
            dVar_dn = A_T.dot(-weights / (t * t))
            weighted_sum = np.dot(s, dVar_dn)
            return total_shots * s * (dVar_dn - weighted_sum)

        if warm_start is not None:
            x0_n = np.maximum(warm_start, 0.001)
            x0_n *= total_shots / x0_n.sum()
        else:
            x0_n = np.full(num_groups, total_shots / num_groups)

        theta0 = np.log(x0_n / x0_n.min())

        result = minimize(
            objective_theta, theta0, method='L-BFGS-B', jac=gradient_theta,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': verbose})

        if not result.success:
            import warnings
            warnings.warn(f"Optimization did not converge: {result.message}")

        optimal_shots = total_shots * softmax(result.x)
        optimal_variance = objective_theta(result.x)
        return optimal_shots, optimal_variance

    @classmethod
    def load_symplectic(cls, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        from ogn.pauli import PauliString
        collection = cls()
        n_qubits = 0
        for group_idx in range(data['num_groups']):
            for x, z in zip(data['x_bits'][group_idx], data['z_bits'][group_idx]):
                support = int(x) | int(z)
                if support > 0:
                    n_qubits = max(n_qubits, support.bit_length())
        if n_qubits == 0:
            n_qubits = 1
        for group_idx in range(data['num_groups']):
            group = PauliGroup()
            for x, z, coeff in zip(data['x_bits'][group_idx], data['z_bits'][group_idx],
                                    data['coefficients'][group_idx]):
                coeff_val = float(np.real(coeff)) if abs(np.imag(coeff)) < 1e-15 else complex(coeff)
                group.add(PauliString(int(x), int(z), coeff_val, n_qubits))
            collection.groups.append(group)
        return collection
