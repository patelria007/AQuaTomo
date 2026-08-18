"""Verify exact Pauli reconstruction and the internal MLE invariants.

Run from the repository root with::

    python verification/3_reconstruction/exact_reconstruction_verification.py

The exact measurement data are constructed with an independent tensor-product
projector oracle.  This is intentional: using ``exact_pauli_measurements`` here
would let the measurement and reconstruction implementations share the same
qubit-ordering or outcome-sign bug.
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

from nbqst.denoise import project_density_matrix  # noqa: E402
from nbqst.measurements import (  # noqa: E402
    MeasurementData,
    complete_pauli_settings,
    simulate_pauli_measurements,
)
from nbqst.metrics import fidelity, hilbert_schmidt_distance  # noqa: E402
from nbqst.reconstruction import (  # noqa: E402
    _likelihood_gradient,
    factorized_mle,
    linear_inversion_pauli,
    negative_log_likelihood,
)


DEFAULT_OUTPUT = Path(__file__).with_name("exact_reconstruction_verification.png")
EXACT_TOLERANCE = 2e-10
PHYSICAL_TOLERANCE = 2e-10
GRADIENT_TOLERANCE = 2e-6
ORACLE_SHOTS = 1_000_000

PAULI = {
    "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
}
IDENTITY = np.eye(2, dtype=np.complex128)


@dataclass(frozen=True)
class ExactCase:
    """One independently measured state used for exact inversion."""

    name: str
    rho: np.ndarray


def density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """Return ``|psi><psi|`` after normalizing the supplied vector."""
    state_vector = np.asarray(state_vector, dtype=np.complex128)
    state_vector = state_vector / np.linalg.norm(state_vector)
    return np.outer(state_vector, state_vector.conj())


def random_density_matrix(dimension: int, seed: int) -> np.ndarray:
    """Create a reproducible full-rank complex density matrix."""
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(dimension, dimension)) + 1j * generator.normal(
        size=(dimension, dimension)
    )
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


def kron_all(factors: list[np.ndarray]) -> np.ndarray:
    """Kronecker product in leftmost-qubit-most-significant order."""
    result = np.asarray([[1.0]], dtype=np.complex128)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def projector_probabilities(rho: np.ndarray, setting: str) -> np.ndarray:
    """Evaluate Born probabilities from explicit Pauli eigenprojectors."""
    probabilities = []
    for outcome in range(2 ** len(setting)):
        bits = format(outcome, f"0{len(setting)}b")
        projectors = [
            (IDENTITY + (1.0 if bit == "0" else -1.0) * PAULI[axis]) / 2.0
            for axis, bit in zip(setting, bits)
        ]
        probabilities.append(float(np.trace(rho @ kron_all(projectors)).real))
    probabilities = np.asarray(probabilities)
    probabilities[np.abs(probabilities) < 5e-15] = 0.0
    return probabilities / probabilities.sum()


def independent_exact_measurements(
    rho: np.ndarray, *, shots_per_setting: int = ORACLE_SHOTS
) -> MeasurementData:
    """Build deterministic probability-weighted counts without production rotations."""
    n_qubits = rho.shape[0].bit_length() - 1
    counts = {
        setting: projector_probabilities(rho, setting) * shots_per_setting
        for setting in complete_pauli_settings(n_qubits)
    }
    return MeasurementData(n_qubits, counts, shots_per_setting)


def exact_cases() -> tuple[ExactCase, ...]:
    """Return cases that exercise complex phases, correlations, and mixed spectra."""
    plus_i = np.asarray([1.0, 1.0j]) / np.sqrt(2.0)
    bell = np.asarray([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    ghz = np.zeros(8, dtype=np.complex128)
    ghz[[0, 7]] = 1.0 / np.sqrt(2.0)
    return (
        ExactCase("one-qubit +i", density_matrix(plus_i)),
        ExactCase("two-qubit Bell", density_matrix(bell)),
        ExactCase("two-qubit mixed", random_density_matrix(4, 20260821)),
        ExactCase("three-qubit GHZ", density_matrix(ghz)),
        ExactCase("three-qubit mixed", random_density_matrix(8, 20260822)),
    )


def density_diagnostics(rho: np.ndarray) -> dict[str, float]:
    """Return the three defining density-matrix residuals."""
    rho = np.asarray(rho)
    hermitian_error = float(np.linalg.norm(rho - rho.conj().T))
    trace_error = float(abs(np.trace(rho) - 1.0))
    minimum_eigenvalue = float(
        np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min().real
    )
    return {
        "hermitian_error": hermitian_error,
        "trace_error": trace_error,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def assert_physical(name: str, rho: np.ndarray) -> dict[str, float]:
    """Assert Hermiticity, unit trace, and positive semidefiniteness."""
    diagnostics = density_diagnostics(rho)
    if diagnostics["hermitian_error"] > PHYSICAL_TOLERANCE:
        raise AssertionError(f"{name}: Hermitian residual is too large")
    if diagnostics["trace_error"] > PHYSICAL_TOLERANCE:
        raise AssertionError(f"{name}: trace residual is too large")
    if diagnostics["minimum_eigenvalue"] < -PHYSICAL_TOLERANCE:
        raise AssertionError(f"{name}: estimate is not positive semidefinite")
    return diagnostics


def verify_exact_inversion() -> tuple[dict[str, object], ...]:
    """Compare production linear inversion with the independent exact oracle."""
    rows = []
    for case in exact_cases():
        data = independent_exact_measurements(case.rho)
        estimate = np.asarray(linear_inversion_pauli(data))
        error = float(np.linalg.norm(estimate - case.rho))
        maximum_error = float(np.max(np.abs(estimate - case.rho)))
        if error > EXACT_TOLERANCE:
            raise AssertionError(
                f"{case.name}: exact inversion error {error:.3e} exceeds "
                f"{EXACT_TOLERANCE:.1e}"
            )
        rows.append(
            {
                "name": case.name,
                "rho": case.rho,
                "estimate": estimate,
                "frobenius_error": error,
                "maximum_error": maximum_error,
            }
        )

    full = independent_exact_measurements(exact_cases()[1].rho)
    missing_setting = next(iter(full.counts))
    incomplete_counts = {
        setting: counts
        for setting, counts in full.counts.items()
        if setting != missing_setting
    }
    incomplete = MeasurementData(full.n_qubits, incomplete_counts, full.shots_per_setting)
    try:
        linear_inversion_pauli(incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError("linear inversion accepted informationally incomplete data")
    return tuple(rows)


def verify_gradient() -> dict[str, float]:
    """Compare the analytic likelihood gradient with a central difference."""
    truth = random_density_matrix(4, 20260823)
    rho = random_density_matrix(4, 20260824)
    data = independent_exact_measurements(truth, shots_per_setting=20_000)
    generator = np.random.default_rng(20260826)
    raw = generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
    direction = 0.5 * (raw + raw.conj().T)
    direction -= np.trace(direction).real * np.eye(4) / 4.0
    direction /= np.linalg.norm(direction)

    epsilon = 2e-7
    gradient = np.asarray(_likelihood_gradient(rho, data, np, 1e-14))
    analytic = float(np.trace(gradient @ direction).real)
    numerical = (
        negative_log_likelihood(rho + epsilon * direction, data, epsilon=1e-14)
        - negative_log_likelihood(rho - epsilon * direction, data, epsilon=1e-14)
    ) / (2.0 * epsilon)
    scale = max(1.0, abs(analytic), abs(numerical))
    relative_error = abs(analytic - numerical) / scale
    if relative_error > GRADIENT_TOLERANCE:
        raise AssertionError(
            f"likelihood gradient relative error {relative_error:.3e} exceeds "
            f"{GRADIENT_TOLERANCE:.1e}"
        )
    return {
        "analytic": analytic,
        "numerical": numerical,
        "relative_error": relative_error,
    }


def verify_mle() -> dict[str, object]:
    """Exercise MLE monotonicity, physicality, accuracy, and rank capping."""
    bell = exact_cases()[1].rho
    data = simulate_pauli_measurements(bell, 512, rng=20260825)
    linear = np.asarray(linear_inversion_pauli(data))
    initial = np.asarray(project_density_matrix(linear))
    estimate, history = factorized_mle(
        data,
        initial=initial,
        max_iter=120,
        learning_rate=0.25,
        return_history=True,
    )
    estimate = np.asarray(estimate)
    diagnostics = assert_physical("full-rank MLE", estimate)
    if any(next_value > value + 2e-12 for value, next_value in zip(history, history[1:])):
        raise AssertionError("MLE accepted a step that increased negative log likelihood")
    if history[-1] > history[0] + 2e-12:
        raise AssertionError("MLE final objective is worse than its initial objective")
    mle_fidelity = float(fidelity(bell, estimate))
    if mle_fidelity < 0.95:
        raise AssertionError(f"Bell-state MLE fidelity is unexpectedly low: {mle_fidelity:.6f}")

    rank_one, rank_history = factorized_mle(
        data,
        initial=initial,
        rank=1,
        max_iter=120,
        learning_rate=0.25,
        return_history=True,
    )
    rank_one = np.asarray(rank_one)
    rank_diagnostics = assert_physical("rank-one MLE", rank_one)
    eigenvalues = np.linalg.eigvalsh(rank_one)
    numerical_rank = int(np.count_nonzero(eigenvalues > 1e-9))
    if numerical_rank > 1:
        raise AssertionError(f"rank-one MLE returned numerical rank {numerical_rank}")
    if any(
        next_value > value + 2e-12
        for value, next_value in zip(rank_history, rank_history[1:])
    ):
        raise AssertionError("rank-one MLE objective is not monotone")

    return {
        "truth": bell,
        "linear": linear,
        "initial": initial,
        "estimate": estimate,
        "history": np.asarray(history),
        "fidelity": mle_fidelity,
        "hs_distance": float(hilbert_schmidt_distance(bell, estimate)),
        "diagnostics": diagnostics,
        "rank_one": rank_one,
        "rank_history": np.asarray(rank_history),
        "rank_one_fidelity": float(fidelity(bell, rank_one)),
        "rank_diagnostics": rank_diagnostics,
    }


def run_verification() -> dict[str, object]:
    """Run every deterministic assertion before any figure is written."""
    return {
        "exact": verify_exact_inversion(),
        "gradient": verify_gradient(),
        "mle": verify_mle(),
    }


def print_report(results: dict[str, object]) -> None:
    """Print the numerical evidence behind the figure."""
    print("Independent-oracle exact linear inversion")
    print(f"{'case':<24} {'Frobenius error':>18} {'max element error':>19}")
    print("-" * 65)
    for row in results["exact"]:
        print(
            f"{row['name']:<24} {row['frobenius_error']:18.3e} "
            f"{row['maximum_error']:19.3e}"
        )

    gradient = results["gradient"]
    print("\nLikelihood-gradient central-difference check")
    print(f"  analytic derivative : {gradient['analytic']:+.10e}")
    print(f"  numerical derivative: {gradient['numerical']:+.10e}")
    print(f"  relative error      : {gradient['relative_error']:.3e}")

    mle = results["mle"]
    print("\nFinite-shot Bell-state MLE")
    print(f"  iterations          : {len(mle['history']) - 1}")
    print(f"  NLL                 : {mle['history'][0]:.9f} -> {mle['history'][-1]:.9f}")
    print(f"  fidelity            : {mle['fidelity']:.8f}")
    print(f"  minimum eigenvalue  : {mle['diagnostics']['minimum_eigenvalue']:+.3e}")
    print(f"  rank-one fidelity   : {mle['rank_one_fidelity']:.8f}")
    print("\nAll exact-reconstruction and MLE-invariant assertions passed.")


def plot_results(results: dict[str, object], output: Path, *, show: bool = False) -> None:
    """Plot elementwise agreement, exact errors, and MLE convergence."""
    exact = results["exact"]
    truth_elements = np.concatenate([row["rho"].ravel() for row in exact])
    estimate_elements = np.concatenate([row["estimate"].ravel() for row in exact])
    mle = results["mle"]

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.4), constrained_layout=True)
    components = (
        (axes[0, 0], truth_elements.real, estimate_elements.real, "Real parts"),
        (axes[0, 1], truth_elements.imag, estimate_elements.imag, "Imaginary parts"),
    )
    for axis, reference, actual, title in components:
        lower = min(float(reference.min()), float(actual.min()))
        upper = max(float(reference.max()), float(actual.max()))
        padding = max(0.02, 0.05 * (upper - lower))
        axis.scatter(reference, actual, s=16, alpha=0.65, color="#3569a8")
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            linestyle="--",
            color="0.25",
            linewidth=1.2,
            label="exact agreement",
        )
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlabel("independent-oracle target")
        axis.set_ylabel("linear reconstruction")
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(loc="upper left")

    error_axis = axes[1, 0]
    names = [row["name"] for row in exact]
    errors = [max(row["frobenius_error"], 1e-17) for row in exact]
    positions = np.arange(len(names))
    error_axis.bar(positions, errors, color="#cf6a32")
    error_axis.axhline(EXACT_TOLERANCE, color="#9b2c2c", linestyle="--", label="acceptance limit")
    error_axis.set_yscale("log")
    error_axis.set_xticks(positions, names, rotation=25, ha="right")
    error_axis.set_ylabel(r"$\|\hat\rho-\rho\|_F$")
    error_axis.set_title("Exact inversion residuals")
    error_axis.grid(axis="y", alpha=0.25)
    error_axis.legend()

    convergence_axis = axes[1, 1]
    history = mle["history"]
    rank_history = mle["rank_history"]
    convergence_axis.plot(
        np.arange(len(history)), history, marker="o", markersize=3, label="full-rank MLE"
    )
    convergence_axis.plot(
        np.arange(len(rank_history)),
        rank_history,
        marker="s",
        markersize=3,
        label="rank-one MLE",
    )
    convergence_axis.set_xlabel("accepted iteration")
    convergence_axis.set_ylabel("normalized negative log likelihood")
    convergence_axis.set_title(
        "Monotone MLE convergence\n"
        f"F(full)={mle['fidelity']:.5f}, F(rank 1)={mle['rank_one_fidelity']:.5f}"
    )
    convergence_axis.grid(alpha=0.25)
    convergence_axis.legend()

    figure.suptitle("Exact reconstruction and MLE invariant verification", fontsize=15)
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
