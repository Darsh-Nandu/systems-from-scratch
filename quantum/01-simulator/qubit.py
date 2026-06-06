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
    def probabilites(self) -> np.ndarray:
        """ Returns: [P(|0>), P(|1>)]"""
        return np.abs(self.state)** 2

    def measure(self) -> int:
        """
        Process :-
        1. Compute outcome probabilites vis the Born rule.
        2. sample a classical outcome (0 or 1)
        3. Collapse the statevector to the corresponding basis state.
        4. Return the classical outcome.

        This is a destructive operation: after calling measure(), the
        qubit is in a definite classical state - all quantum information
        in the original superposition is irreversibly lost.
        """
        probs = self.probabilites
        outcome = int(np.random.choice([0, 1], p=probs))
        self.state = np.array([1.0, 0,0] if outcome == 0 else [0.0, 1.0], dtype=complex)

        return outcome

    def __repr__(self) -> str:
        a, b = self.alpha, self.beta
        p0, p1 = self.probabilites
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