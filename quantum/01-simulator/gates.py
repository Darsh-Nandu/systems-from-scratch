"""
gates.py - quantum gates as unitary matrices.
"""

import numpy as np
from qubit import Qubit

def X() -> np.ndarray:
    """
    Pauli-X gate - quantum NOT.
    180° rotation atrind the X-axis of the Bloch sphere.

    X|0⟩ = |1⟩,  X|1⟩ = |0⟩
    """
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)

def Y() -> np.ndarray:
    """
    Pauli-Y gate.
    180° rotation around the Y-axis.

    Y|0⟩ = i|1⟩,  Y|1⟩ = -i|0⟩
    """
    return np.array([[0, -1j],
                     [1j, 0]], dtype=complex)

def Z() -> np.ndarray:
    """
    Pauli-Z gate - phase flip.
    180° rotation around the Z-axis.

    Z|0⟩ = |0⟩,  Z|1⟩ = -|1⟩
    """
    return np.array([[1, 0],
                     [0, -1]], dtype=complex)

def H() -> np.ndarray:
    """
    Hadamard gate - superposition engine.
    180° rotation around the axis bisecting X and Z.

    H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
    H|1⟩ = |−⟩ = (|0⟩ - |1⟩)/√2

    H is its own inverse: H² = I
    """
    return (1/np.sqrt(2)) * np.array([[1, 1],
                                      [1, -1]], dtype=complex)

def S() -> np.ndarray:
    """
    Phase gate S - 90° rotation around Z-axis
    Applies a phase of i (= e^{iπ/2}) to |1⟩, leaves |0⟩ unchanged
    """
    return np.array([[1, 0],
                     [0, 1j]], dtype=complex)

def T() -> np.ndarray:
    """
    T gate (π/8 gate) — 45° rotation around Z-axis.
    Applies a phase of e^{iπ/4} to |1⟩.

    Together with H, the T gate is universal for single-qubit computation:
    any single-qubit unitary can be approximated to arbitrary precision
    using only H and T (Solovay-Kitaev theorem).
    """
    return np.array([[1, 0],
                     [0, np.exp(1j * np.pi / 4)]], dtype=complex)

def Rx(theta: float) -> np.ndarray:
    """
    Rotation by angle theta around the X-axis of the Bloch sphere.

    Rx(θ) = [[cos(θ/2), -i.sin(θ/2)],
             [-i.sin(θ/2), cos(θ/2)]]
    
    Parameters:
    theta: angle in radians
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j*s],
                     [-1j*s, c]], dtype=complex)

def Ry(theta: float) -> np.ndarray:
    """
    Rotation by angle theta around the Y-axis of the Bloch sphere.

    Ry(θ) = [[cos(θ/2),  -sin(θ/2)],
              [sin(θ/2),   cos(θ/2)]]
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s],
                     [s, c]], dtype=complex)

def Rz(theta: float) -> np.ndarray:
    """
    Rotation by angle theta around the Z-axis of the Bloch sphere.

    Rz(θ) = [[e^{-iθ/2},    0      ],
              [0,          e^{iθ/2}]]
    """
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

def apply(gate: np.ndarray, qubit: Qubit) -> Qubit:
    """
    Apply a unitary gate matrix to a qubit's statevector.
    """
    _assert_unitary(gate)
    qubit.state = gate @ qubit.state
    return qubit

def _assert_unitary(U: np.ndarray, tol: float = 1e-9):
    should_be_identity = np.conj(U).T @ U
    identity = np.eye(U.shape[0], dtype=complex)
    if not np.allclose(should_be_identity, identity, atol = tol):
        raise ValueError(
            f"Gate matrix is not unitary.\n"
            f"U†U =\n{should_be_identity}\n"
            f"Expected identity matrix."
        )

def compose(*gates: np.ndarray) -> np.ndarray:
    """
    Compose multiple gates into a single unitary matrix.

    Gates are applied left-to-right in the argument order — so
    compose(H(), X(), H()) applies H first, then X, then H.

    Mathematically this reverses the matrix multiplication order:
        U_total = U_last @ ... @ U_first

    This matches circuit diagram convention: gates drawn left to right
    correspond to matrices multiplied right to left.
    """
    result = np.eye(2, dtype=complex)
    for gate in reversed(gates):
        result = gate @ result
    return result

def display(gate: np.ndarray, name: str = "Gate"):
    """Print a gate matrix in readble format."""
    print(f"\n{name}:")
    for row in gate:
        formatted = "  ".join(f"{v.real:+.4f}{v.imag:+.4f}i" for v in row)
        print(f"  [ {formatted} ]")
    print()

# Smoke tests

if __name__ == "__main__":
    from qubit import ket_zero, ket_one, ket_plus

    print("=== Gate matrix: Hadamard ===")
    display(H(), "H")

    print("=== X|0⟩ should give |1⟩ ===")
    q = ket_zero()
    apply(X(), q)
    print(q)

    print("=== H|0⟩ should give |+⟩ (P0=P1=0.5) ===")
    q = ket_zero()
    apply(H(), q)
    print(q)

    print("=== H|1⟩ should give |−⟩ (P0=P1=0.5, but phase differs) ===")
    q = ket_one()
    apply(H(), q)
    print(q)

    print("=== H² = I: applying H twice returns to |0⟩ ===")
    q = ket_zero()
    apply(H(), q)
    apply(H(), q)
    print(q)

    print("=== Composition: HXH = Z (conjugation identity) ===")
    hxh = compose(H(), X(), H())
    print("HXH == Z?", np.allclose(hxh, Z()))

    print("=== Phase matters: Z|+⟩ = |−⟩, same probs but different state ===")
    q_plus  = ket_plus()
    q_minus = ket_plus()
    apply(Z(), q_minus)
    print("Z|+⟩:", q_minus)
    print("Same probs as |+⟩?", np.allclose(q_plus.probabilities,
                                              q_minus.probabilities))
    print("Same state as |+⟩?", np.allclose(q_plus.state, q_minus.state))

    print("\n=== Invalid gate should raise ===")
    try:
        bad_gate = np.array([[2, 0], [0, 1]], dtype=complex)
        apply(bad_gate, ket_zero())
    except ValueError as e:
        print(f"Caught: {e}")