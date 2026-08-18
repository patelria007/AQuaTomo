"""Verify and visualize Born probabilities for local Pauli measurements.

Run from the repository root with::

    python verification/measurement_simulation/born_probability_verification.py

The implementation is compared with an independent projector-based oracle.
Numerical assertions run before the explanatory figure is written.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nbqst.measurements import (  # noqa: E402
    complete_pauli_settings,
    pauli_probabilities,
)


DEFAULT_OUTPUT = Path(__file__).with_name("born_probability_verification.png")
TOLERANCE = 2e-12
RANDOM_SEED = 20260818

PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
}


@dataclass(frozen=True)
class ReferenceCase:
    """One state/setting pair with analytically known probabilities."""

    name: str
    rho: np.ndarray
    setting: str
    expected: np.ndarray


def density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """Return the pure-state density matrix |psi><psi|."""
    state_vector = np.asarray(state_vector, dtype=np.complex128)
    return np.outer(state_vector, state_vector.conj())


def kron_all(factors: list[np.ndarray]) -> np.ndarray:
    """Kronecker product with the leftmost qubit as the most significant."""
    result = np.asarray([[1.0]], dtype=np.complex128)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def projector_oracle(rho: np.ndarray, setting: str) -> np.ndarray:
    """Compute Born probabilities from tensor-product eigenprojectors.

    This deliberately does not use the production basis-rotation matrices.
    Outcome bit 0/1 selects the +1/-1 Pauli eigenprojector, respectively.
    """
    identity = PAULI["I"]
    probabilities = []
    for outcome in range(2 ** len(setting)):
        bits = format(outcome, f"0{len(setting)}b")
        local_projectors = [
            (identity + (1 if bit == "0" else -1) * PAULI[axis]) / 2.0
            for axis, bit in zip(setting, bits)
        ]
        projector = kron_all(local_projectors)
        probabilities.append(np.trace(np.asarray(rho) @ projector).real)
    return np.asarray(probabilities)


def reference_cases() -> tuple[ReferenceCase, ...]:
    """Return one- and two-qubit cases with known exact distributions."""
    zero = np.asarray([1, 0], dtype=np.complex128)
    plus = np.asarray([1, 1], dtype=np.complex128) / np.sqrt(2.0)
    plus_y = np.asarray([1, 1j], dtype=np.complex128) / np.sqrt(2.0)
    zero_zero = np.kron(zero, zero)
    bell = np.asarray([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2.0)

    return (
        ReferenceCase("|0>, Z", density_matrix(zero), "Z", np.asarray([1, 0])),
        ReferenceCase("|0>, X", density_matrix(zero), "X", np.asarray([0.5, 0.5])),
        ReferenceCase("|+>, X", density_matrix(plus), "X", np.asarray([1, 0])),
        ReferenceCase("|+i>, Y", density_matrix(plus_y), "Y", np.asarray([1, 0])),
        ReferenceCase("I/2, Y", np.eye(2) / 2.0, "Y", np.asarray([0.5, 0.5])),
        ReferenceCase(
            "|00>, ZZ",
            density_matrix(zero_zero),
            "ZZ",
            np.asarray([1, 0, 0, 0]),
        ),
        ReferenceCase(
            "Bell, XX",
            density_matrix(bell),
            "XX",
            np.asarray([0.5, 0, 0, 0.5]),
        ),
        ReferenceCase(
            "Bell, YY",
            density_matrix(bell),
            "YY",
            np.asarray([0, 0.5, 0.5, 0]),
        ),
        ReferenceCase(
            "I/4, XY",
            np.eye(4) / 4.0,
            "XY",
            np.full(4, 0.25),
        ),
    )


def random_density_matrix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    """Generate a reproducible full-rank complex density matrix."""
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


def compatible_expectations(
    rho: np.ndarray, setting: str, probabilities: np.ndarray
) -> tuple[list[float], list[float], list[str]]:
    """Compare all nontrivial compatible Pauli expectations."""
    born_values: list[float] = []
    trace_values: list[float] = []
    labels: list[str] = []
    n_qubits = len(setting)

    for support in itertools.product((0, 1), repeat=n_qubits):
        if not any(support):
            continue
        label = "".join(axis if used else "I" for axis, used in zip(setting, support))
        signs = []
        for outcome in range(2**n_qubits):
            bits = format(outcome, f"0{n_qubits}b")
            sign = np.prod(
                [(-1 if bit == "1" else 1) for bit, used in zip(bits, support) if used]
            )
            signs.append(sign)
        born_values.append(float(np.dot(signs, probabilities)))
        operator = kron_all([PAULI[axis] for axis in label])
        trace_values.append(float(np.trace(rho @ operator).real))
        labels.append(label)

    return born_values, trace_values, labels


def run_verification() -> dict[str, object]:
    """Execute deterministic reference, oracle, and consistency checks."""
    cases = reference_cases()
    known_rows = []
    for case in cases:
        actual = np.asarray(pauli_probabilities(case.rho, case.setting), dtype=float)
        oracle = projector_oracle(case.rho, case.setting)
        if not np.allclose(actual, case.expected, atol=TOLERANCE, rtol=0.0):
            raise AssertionError(f"{case.name}: production result differs from analytic result")
        if not np.allclose(oracle, case.expected, atol=TOLERANCE, rtol=0.0):
            raise AssertionError(f"{case.name}: projector oracle differs from analytic result")
        known_rows.append((case, actual))

    rng = np.random.default_rng(RANDOM_SEED)
    oracle_reference = []
    oracle_actual = []
    expectation_reference = []
    expectation_actual = []
    expectation_labels = []

    for _ in range(8):
        rho = random_density_matrix(rng, 4)
        settings = complete_pauli_settings(2)
        if len(settings) != 3**2 or len(set(settings)) != len(settings):
            raise AssertionError("two-qubit setting enumeration is incomplete or duplicated")
        for setting in settings:
            actual = np.asarray(pauli_probabilities(rho, setting), dtype=float)
            oracle = projector_oracle(rho, setting)
            if np.min(actual) < -TOLERANCE:
                raise AssertionError(f"{setting}: a Born probability is negative")
            if abs(float(actual.sum()) - 1.0) > TOLERANCE:
                raise AssertionError(f"{setting}: Born probabilities do not sum to one")
            if not np.allclose(actual, oracle, atol=TOLERANCE, rtol=0.0):
                raise AssertionError(f"{setting}: rotation and projector calculations disagree")

            oracle_reference.extend(oracle.tolist())
            oracle_actual.extend(actual.tolist())
            born, trace, labels = compatible_expectations(rho, setting, actual)
            expectation_actual.extend(born)
            expectation_reference.extend(trace)
            expectation_labels.extend(labels)

    oracle_reference_array = np.asarray(oracle_reference)
    oracle_actual_array = np.asarray(oracle_actual)
    expectation_reference_array = np.asarray(expectation_reference)
    expectation_actual_array = np.asarray(expectation_actual)
    probability_error = float(np.max(np.abs(oracle_actual_array - oracle_reference_array)))
    expectation_error = float(
        np.max(np.abs(expectation_actual_array - expectation_reference_array))
    )

    if probability_error > TOLERANCE:
        raise AssertionError("maximum projector-oracle error exceeds tolerance")
    if expectation_error > TOLERANCE:
        raise AssertionError("Born probabilities and Pauli traces are inconsistent")

    return {
        "known_rows": tuple(known_rows),
        "oracle_reference": oracle_reference_array,
        "oracle_actual": oracle_actual_array,
        "expectation_reference": expectation_reference_array,
        "expectation_actual": expectation_actual_array,
        "expectation_labels": tuple(expectation_labels),
        "probability_error": probability_error,
        "expectation_error": expectation_error,
    }


def print_report(results: dict[str, object]) -> None:
    """Print the numerical values behind the figure."""
    print("Born-probability verification")
    print(f"{'reference case':<18} {'setting':>8} {'max error':>14} {'sum error':>14}")
    print("-" * 58)
    for case, actual in results["known_rows"]:
        max_error = np.max(np.abs(actual - case.expected))
        sum_error = abs(float(np.sum(actual)) - 1.0)
        print(f"{case.name:<18} {case.setting:>8} {max_error:14.3e} {sum_error:14.3e}")
    print(f"\nRandom 2Q projector-oracle max error: {results['probability_error']:.3e}")
    print(f"Compatible-Pauli expectation max error: {results['expectation_error']:.3e}")
    print("All deterministic Born-probability checks passed.")


def plot_results(results: dict[str, object], output: Path, *, show: bool = False) -> None:
    """Visualize analytic cases and the two independent cross-checks."""
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    selected = [row for row in results["known_rows"] if len(row[0].setting) == 2]
    probability_matrix = np.vstack([actual for _, actual in selected])
    image = axes[0].imshow(probability_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(range(4), ["00", "01", "10", "11"])
    axes[0].set_yticks(range(len(selected)), [case.name for case, _ in selected])
    axes[0].set_xlabel("measurement outcome bit string")
    axes[0].set_title("A. Analytic two-qubit cases")
    for row in range(probability_matrix.shape[0]):
        for column in range(probability_matrix.shape[1]):
            value = probability_matrix[row, column]
            axes[0].text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
            )
    colorbar = figure.colorbar(image, ax=axes[0], shrink=0.78)
    colorbar.set_label("Born probability")

    reference = results["oracle_reference"]
    actual = results["oracle_actual"]
    extent = [min(reference.min(), actual.min()), max(reference.max(), actual.max())]
    padding = max(0.02, 0.05 * (extent[1] - extent[0]))
    line_limits = [extent[0] - padding, extent[1] + padding]
    axes[1].scatter(reference, actual, s=18, alpha=0.55, color="#3569a8")
    axes[1].plot(line_limits, line_limits, color="0.25", linestyle="--", linewidth=1.2)
    axes[1].set_xlim(line_limits)
    axes[1].set_ylim(line_limits)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel(r"projector oracle $\mathrm{Tr}(\rho\Pi_b)$")
    axes[1].set_ylabel(r"implementation $\mathrm{diag}(U^\dagger\rho U)_b$")
    axes[1].set_title("B. Independent Born-rule oracle")
    axes[1].grid(alpha=0.22)

    reference = results["expectation_reference"]
    actual = results["expectation_actual"]
    axes[2].scatter(reference, actual, s=18, alpha=0.5, color="#cf6a32")
    axes[2].plot([-1.04, 1.04], [-1.04, 1.04], color="0.25", linestyle="--", linewidth=1.2)
    axes[2].set_xlim(-1.04, 1.04)
    axes[2].set_ylim(-1.04, 1.04)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel(r"direct trace $\mathrm{Tr}(\rho P)$")
    axes[2].set_ylabel("Born-probability parity sum")
    axes[2].set_title("C. Pauli-expectation consistency")
    axes[2].grid(alpha=0.22)

    figure.suptitle("Local-Pauli Born-probability verification", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_verification()
    print_report(results)
    plot_results(results, args.output, show=args.show)
    print(f"Figure written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
