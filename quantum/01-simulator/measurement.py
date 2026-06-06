"""
measurement.py: Multi-shot sampling and statistics

Real quantum computers (and good simulators) run a circuit many times.
Each run is a 'shot'. This module provides the statistical layer:
sampling, counting, and expectation values over many shots.
"""

import numpy as np
import copy
from collections import Counter
from qubit import Qubit
from gates import apply

def sample(qubit: Qubit, shots: int = 1024, basis: str = 'Z') -> Counter:
    counts = Counter()
    for _ in range(shots):
        q_copy = copy.deepcopy(qubit)
        outcome = q_copy.measure(basis=basis)
        counts[str(outcome)] += 1
    return counts

def marginal_probs(counts: Counter, shots: int) -> dict:
    """
    Converts raw counts to empirical probabilites.
    """
    return {k: v / shots for k, v in counts.items()}

def expectation_value(qubit: Qubit, shots: int = 1024) -> float:
    """
    Estimate ⟨Z⟩ = P(0) - P(1) empirically from shot counts.

    This is the *sampled* estimator — it converges to the exact
    value as shots → ∞, but has statistical noise for finite shots.
    Compare with qubit.expectation_Z for the exact analytic value.
    """
    counts = sample(qubit, shots=shots, basis='Z')
    n0 = counts.get('0',0)
    n1 = counts.get('1',0)
    return (n0 - n1) / shots

def print_results(counts: Counter, shots: int, title: str = "Results"):
    """Pretty-print measurement results as a text histogram"""
    print(f"\n{title} ({shots} shots)")
    print("-" * 40)
    for outcome in sorted(counts):
        count = counts[outcome]
        prob  = count / shots
        bar   = "█" * int(prob * 40)
        print(f"  |{outcome}⟩ : {bar:<40} {count:>5} ({prob:.3f})")
    print()

# Smoke tests

if __name__ == "__main__":
    from qubit import ket_zero, ket_one, ket_plus, ket_minus, Qubit
    from gates import apply, H, Z

    SHOTS = 2048

    # Test 1: |0⟩ always gives 0
    print("=== |0⟩ measured in Z-basis (should be 100% '0') ===")
    counts = sample(ket_zero(), shots=SHOTS)
    print_results(counts, SHOTS, "|0⟩ Z-basis")

    # Test 2: |+⟩ is 50/50 in Z-basis
    print("=== |+⟩ measured in Z-basis (should be ~50/50) ===")
    counts = sample(ket_plus(), shots=SHOTS)
    print_results(counts, SHOTS, "|+⟩ Z-basis")

    # Test 3: |+⟩ always gives 0 in X-basis
    # |+⟩ IS the |0⟩ of the X-basis, so measuring it there is certain
    print("=== |+⟩ measured in X-basis (should be 100% '0') ===")
    counts = sample(ket_plus(), shots=SHOTS, basis='X')
    print_results(counts, SHOTS, "|+⟩ X-basis")

    # Test 4: phase is invisible to Z-measurement
    # |+⟩ and |−⟩ have identical Z-basis distributions
    print("=== |−⟩ measured in Z-basis (same as |+⟩ — phase is hidden) ===")
    counts = sample(ket_minus(), shots=SHOTS)
    print_results(counts, SHOTS, "|−⟩ Z-basis")

    # Test 5: phase IS visible in X-basis
    # |−⟩ is the |1⟩ of the X-basis
    print("=== |−⟩ measured in X-basis (should be 100% '1') ===")
    counts = sample(ket_minus(), shots=SHOTS, basis='X')
    print_results(counts, SHOTS, "|−⟩ X-basis")

    # Test 6: expectation value
    q = Qubit(alpha=0.6, beta=0.8j)
    exact   = q.expectation_Z
    sampled = expectation_value(q, shots=SHOTS)
    print(f"=== ⟨Z⟩ for α=0.6, β=0.8i ===")
    print(f"  Exact  (analytic): {exact:.4f}")
    print(f"  Sampled ({SHOTS} shots): {sampled:.4f}")
    print(f"  Error: {abs(exact - sampled):.4f}  (shrinks as shots → ∞)\n")