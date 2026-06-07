"""
qubit.py - Single qubit representation and measurement.

A qubit is a unit vector in ℂ², the 2D complex vector.
State: |ψ⟩ = α|0⟩ + β|1⟩  where  |α|² + |β|² = 1

Convention:
    statevector[0] = α
    statevector[1] = β
"""

import numpy as np

class Qubit:

    def __init__(self, alpha: complex = 1.0, beta: complex = 0.0):
        self.state = np.array([alpha, beta], dtype=complex)
        self._validate()

    def _validate(self):

        """ Enforce the normalisation constraint: |α|² + |β|² = 1."""

        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 1, atol=1e-9):
            raise ValueError(
                f"Qubit state is not normalized: ‖ψ‖ = {norm:.6f}."
                f"Amplitudes must satisfy |α|² + |β|² = 1."
            )


    @property
    def alpha(self) -> complex:
        """Amplitude for |0⟩."""
        return self.state[0]

    @property
    def beta(self) -> complex:
        """Amplitude for |1⟩."""
        return self.state[1]

    @property
    def probabilities(self) -> np.ndarray:
        """ Returns: [P(|0>), P(|1>)]"""
        return np.abs(self.state)** 2

    def measure(self, basis: str = 'Z') -> int:
        """
        Measure the qubit in given basis.
        basis : str
            'Z' — computational basis {|0⟩, |1⟩}  (default)
            'X' — Hadamard basis {|+⟩, |−⟩}
            'Y' — Y basis {|i+⟩, |i-⟩}
        """
        from gates import apply, H, S

        if basis == 'Z':
            rotated = self.state.copy()
        elif basis == 'X':
            import copy
            temp = copy.deepcopy(self)
            apply(H(), temp)
            rotated = temp.state
        elif basis == 'Y':
            import copy
            temp = copy.deepcopy(self)
            Sdagger = np.conj(S()).T
            apply(H(), temp)
            apply(Sdagger, temp)
            rotated = temp.state
        else:
             raise ValueError(f"Unknown basis '{basis}'. Choose 'X','Y', or 'Z'.")

        probs = np.abs(rotated) ** 2
        outcome = int(np.random.choice([0, 1], p=probs))

        self.state = np.array(
            [1.0, 0.0] if outcome == 0 else [0.0, 1.0],
            dtype = complex
        )
        return outcome

    @property
    def expectation_Z(self) -> float:
        p = self.probabilities
        return float(p[0] - p[1])

    def __repr__(self) -> str:
        a, b = self.alpha, self.beta
        p0, p1 = self.probabilities
        return (
            f"Qubit(\n"
            f"  |ψ⟩ = ({a.real:+.4f}{a.imag:+.4f}i)|0⟩ "
            f"+ ({b.real:+.4f}{b.imag:+.4f}i)|1⟩\n"
            f"  P(0) = {p0:.4f},  P(1) = {p1:.4f}\n"
            f")"
        )


def ket_zero() -> Qubit:
        return Qubit(alpha=1.0, beta=0.0)

def ket_one() -> Qubit:
        return Qubit(alpha=1.0, beta=0.0)

def ket_plus() -> Qubit:
        return Qubit(alpha=1/np.sqrt(2), beta=1/np.sqrt(2))

def ket_minus() -> Qubit:
        return Qubit(alpha=1/np.sqrt(2), beta=-1/np.sqrt(2))

# Smoke tests
"""
if __name__ == "__main__":

    print("=== |0⟩ ===")
    q = ket_zero()
    print(q)
    print(f"  Measured: {q.measure()}\n")

    print("=== |+⟩ (1000 measurements — should be ~50/50) ===")
    results = [ket_plus().measure() for _ in range(1000)]
    print(f"  |0⟩ count: {results.count(0)}")
    print(f"  |1⟩ count: {results.count(1)}\n")

    print("=== Phase state: α=0.6, β=0.8i ===")
    q = Qubit(alpha=0.6, beta=0.8j)
    print(q)

    print("=== Invalid state (should raise) ===")
    try:
        bad = Qubit(alpha=1.0, beta=1.0)
    except ValueError as e:
        print(f"  Caught: {e}")
"""

class QuantumRegister:
    """
    A register of n qubits represented as a single statevector in ℂ^{2ⁿ}.
    
    This is the fundamental multi-qubit object. Rather than tracking
    n separate Qubit objects, we maintain ONE vector of length 2ⁿ -
    because qubits can be entangled and cannot always be separated.
    """

    def __init__(self, n: int, init_state: np.ndarray = None):
        self.n = n
        self.dim = 2 ** n

        if init_state is not None:
            state = np.array(init_state, dtype=complex)
            if state.shape != (self.dim,):
                raise ValueError(
                    f"init_state must have length 2^n = {self.dim}, "
                    f"got {state.shape}"
                )
            norm = np.linalg.norm(state)
            if not np.isclose(norm, 1.0, atol=1e-9):
                raise ValueError(f"init_state must be normalized, got norm={norm:.6f}")
            self.state = state
        else:
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[0] = 1

    @classmethod
    def from_qubits(cls, *qubits: Qubit) -> 'QuantumRegister':
        state = qubits[0].state
        for q in qubits[1:]:
            state = np.kron(state, q.state)
        return cls(len(qubits), init_state = state)


    def apply_gate(self, gate: np.ndarray, target: int):
        """
        Apply a single-qubit gate to qubit `target`, leaving others unchanged.
        """
        if not (0 <= target < self.n):
            raise ValueError(f"Target qubit {target} out of range [0, {self.n-1}]")

        full_gate = np.array([[1.0 + 0j]])

        for i in range(self.n):
            full_gate = np.kron(full_gate, gate if i == target else np.eye(2, dtype=complex))

        self.state = full_gate @ self.state


    def apply_two_qubit_gate(self, gate: np.ndarray, control: int, target: int):
        """
        Apply a 4×4 two-qubit gate to qubits (control, target).

        For now this only handles adjacent qubits or a full 2-qubit register.
        """
        if self.n == 2 and {control, target} == {0, 1}:
            self.state = gate @ self.state
        else:
            raise NotImplementedError("General multi-qubit placement not implimented yet!")


    def measure_qubit(self, target: int) -> int:
        """
        Measure a single qubit in the computational basis.

        Computes the marginal probability of qubit `target` being 0 or 1
        by summing over all amplitudes where that qubit's bit is 0 (or 1).
        Then collapses and renormalises the full statevector.
        """
        # Build masks: which indices have qubit `target` = 0 or 1?
        # For n qubits, qubit `target` corresponds to bit (n-1-target)
        # in the binary index (big-endian convention).
        bit_position = self.n - 1 - target

        indices_0 = [i for i in range(self.dim) if not (i >> bit_position & 1)]
        indices_1 = [i for i in range(self.dim) if      i >> bit_position & 1]

        p0 = float(np.sum(np.abs(self.state[indices_0]) ** 2))
        p1 = float(np.sum(np.abs(self.state[indices_1]) ** 2))

        outcome = int(np.random.choice([0, 1], p=[p0, p1]))

        # Collapse
        kill = indices_1 if outcome == 0 else indices_0
        self.state[kill] = 0.0

        # Renormalize
        norm = np.linalg.norm(self.state)
        self.state /= norm

    def measure_all(self) -> list[int]:
        """
        Measure all qubits sequentially. Returns list of classical bits.
        """
        return [self.measure_qubit(i) for i in range(self.n)]

    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def __repr__(self) -> str:
        lines = [f"QuantumRegister({self.n} qubits, dim={self.dim})"]
        for i, amp in enumerate(self.state):
            if abs(amp) > 1e-9:           # skip near-zero amplitudes
                bits = format(i, f'0{self.n}b')
                prob = abs(amp) ** 2
                lines.append(
                    f"  |{bits}⟩ : ({amp.real:+.4f}{amp.imag:+.4f}i)"
                    f"  P={prob:.4f}"
                )
        return "\n".join(lines)

# Smoke tests

if __name__ == "__main__":
    from qubit import ket_zero, ket_one, ket_plus
    from gates import H, X

    print("=== |00⟩ — default 2-qubit register ===")
    reg = QuantumRegister(2)
    print(reg)

    print("\n=== |01⟩ — tensor of |0⟩ and |1⟩ ===")
    reg = QuantumRegister.from_qubits(ket_zero(), ket_one())
    print(reg)

    print("\n=== Apply H to qubit 0 of |00⟩ → (|00⟩ + |10⟩)/√2 ===")
    reg = QuantumRegister(2)
    reg.apply_gate(H(), target=0)
    print(reg)

    print("\n=== Apply H to qubit 1 of |00⟩ → (|00⟩ + |01⟩)/√2 ===")
    reg = QuantumRegister(2)
    reg.apply_gate(H(), target=1)
    print(reg)

    print("\n=== Measure qubit 0 of (|00⟩ + |10⟩)/√2 — should collapse ===")
    reg = QuantumRegister(2)
    reg.apply_gate(H(), target=0)
    print("Before:", reg)
    outcome = reg.measure_qubit(0)
    print(f"Measured qubit 0: {outcome}")
    print("After:", reg)