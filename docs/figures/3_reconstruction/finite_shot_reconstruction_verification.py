"""Verify finite-shot reconstruction accuracy, scaling, and physicality.

Run from the repository root with::

    python verification/3_reconstruction/finite_shot_reconstruction_verification.py

The experiment uses fixed target states and independent seeded multinomial
datasets.  Assertions are evaluated over ensembles rather than requiring every
individual noisy realization to improve monotonically with shot count.
"""

from __future__ import annotations

import argparse
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
from nbqst.measurements import simulate_pauli_measurements  # noqa: E402
from nbqst.metrics import fidelity, hilbert_schmidt_distance  # noqa: E402
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli  # noqa: E402

from exact_reconstruction_verification import (  # noqa: E402
    PHYSICAL_TOLERANCE,
    assert_physical,
    density_diagnostics,
    density_matrix,
    random_density_matrix,
)


SHOT_COUNTS = (64, 256, 1024, 4096)
DEFAULT_TRIALS = 20
DEFAULT_SEED = 20260830
DEFAULT_OUTPUT = Path(__file__).with_name(
    "finite_shot_reconstruction_verification.png"
)
MLE_MAX_ITERATIONS = 80
SCALING_SLOPE_TARGET = -0.5
SCALING_SLOPE_TOLERANCE = 0.13


@dataclass(frozen=True)
class TargetState:
    """A fixed two-qubit target used at every shot count."""

    name: str
    rho: np.ndarray


def target_states() -> tuple[TargetState, ...]:
    """Return product, entangled, and full-rank mixed targets."""
    first = np.asarray([np.cos(0.37), np.exp(0.41j) * np.sin(0.37)])
    second = np.asarray([np.cos(0.63), np.exp(-0.29j) * np.sin(0.63)])
    product = np.kron(first, second)
    bell = np.asarray([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    return (
        TargetState("product pure", density_matrix(product)),
        TargetState("Bell entangled", density_matrix(bell)),
        TargetState("full-rank mixed", random_density_matrix(4, 20260831)),
    )


def collect_trials(trials: int, seed: int) -> dict[str, object]:
    """Reconstruct all target, shot-count, and seed combinations."""
    if trials < 8:
        raise ValueError("trials must be at least 8 for the ensemble checks")

    targets = target_states()
    generator = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    representative_history = None

    for target_index, target in enumerate(targets):
        for shots in SHOT_COUNTS:
            for trial in range(trials):
                data = simulate_pauli_measurements(
                    target.rho,
                    shots,
                    rng=generator,
                )
                if any(int(np.sum(counts)) != shots for counts in data.counts.values()):
                    raise AssertionError("a measurement setting did not conserve shots")

                linear = np.asarray(linear_inversion_pauli(data))
                projected = np.asarray(project_density_matrix(linear))
                mle, history = factorized_mle(
                    data,
                    initial=projected,
                    max_iter=MLE_MAX_ITERATIONS,
                    learning_rate=0.25,
                    return_history=True,
                )
                mle = np.asarray(mle)
                history = np.asarray(history)

                linear_diagnostics = density_diagnostics(linear)
                projected_diagnostics = assert_physical("projected estimate", projected)
                mle_diagnostics = assert_physical("MLE estimate", mle)
                if linear_diagnostics["hermitian_error"] > PHYSICAL_TOLERANCE:
                    raise AssertionError("linear inversion lost Hermiticity")
                if linear_diagnostics["trace_error"] > PHYSICAL_TOLERANCE:
                    raise AssertionError("linear inversion lost unit trace")
                if np.any(np.diff(history) > 2e-12):
                    raise AssertionError("MLE accepted a likelihood-increasing iteration")
                if history[-1] > history[0] + 2e-12:
                    raise AssertionError("MLE ended above its initial objective")

                estimates = {
                    "linear": linear,
                    "projected": projected,
                    "mle": mle,
                }
                for method, estimate in estimates.items():
                    row = {
                        "target_index": target_index,
                        "target": target.name,
                        "shots": shots,
                        "trial": trial,
                        "method": method,
                        "hs_distance": float(
                            hilbert_schmidt_distance(target.rho, estimate)
                        ),
                        "minimum_eigenvalue": density_diagnostics(estimate)[
                            "minimum_eigenvalue"
                        ],
                    }
                    if method != "linear":
                        row["infidelity"] = 1.0 - float(fidelity(target.rho, estimate))
                    records.append(row)

                records[-1]["iterations"] = len(history) - 1
                records[-1]["nll_improvement"] = float(history[0] - history[-1])
                records[-1]["projected_minimum_eigenvalue"] = projected_diagnostics[
                    "minimum_eigenvalue"
                ]
                records[-1]["mle_minimum_eigenvalue"] = mle_diagnostics[
                    "minimum_eigenvalue"
                ]

                if target.name == "Bell entangled" and shots == 256 and trial == 0:
                    representative_history = history

    if representative_history is None:
        raise AssertionError("the representative MLE history was not collected")
    return {
        "targets": targets,
        "records": records,
        "trials": trials,
        "seed": seed,
        "representative_history": representative_history,
    }


def rows_for(
    records: list[dict[str, object]],
    *,
    method: str | None = None,
    target: str | None = None,
    shots: int | None = None,
) -> list[dict[str, object]]:
    """Filter flat experiment records by selected keys."""
    return [
        row
        for row in records
        if (method is None or row["method"] == method)
        and (target is None or row["target"] == target)
        and (shots is None or row["shots"] == shots)
    ]


def root_mean_square(values: list[float]) -> float:
    """Return the square root of the mean squared value."""
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def analyze_trials(data: dict[str, object]) -> dict[str, object]:
    """Aggregate ensemble diagnostics and enforce statistical acceptance checks."""
    records = data["records"]
    targets = data["targets"]
    rms_hs: dict[str, np.ndarray] = {}
    for method in ("linear", "projected", "mle"):
        rms_hs[method] = np.asarray(
            [
                root_mean_square(
                    [
                        row["hs_distance"]
                        for row in rows_for(records, method=method, shots=shots)
                    ]
                )
                for shots in SHOT_COUNTS
            ]
        )

    linear_slope = float(
        np.polyfit(np.log(SHOT_COUNTS), np.log(rms_hs["linear"]), 1)[0]
    )
    if abs(linear_slope - SCALING_SLOPE_TARGET) > SCALING_SLOPE_TOLERANCE:
        raise AssertionError(
            f"linear-inversion RMSE slope {linear_slope:.4f} is inconsistent with "
            f"the inverse-square-root shot law"
        )

    mean_infidelity: dict[tuple[str, str], np.ndarray] = {}
    for target in targets:
        for method in ("projected", "mle"):
            values = np.asarray(
                [
                    np.mean(
                        [
                            row["infidelity"]
                            for row in rows_for(
                                records,
                                method=method,
                                target=target.name,
                                shots=shots,
                            )
                        ]
                    )
                    for shots in SHOT_COUNTS
                ]
            )
            if values[-1] >= values[0]:
                raise AssertionError(
                    f"{target.name} {method}: mean infidelity did not improve from "
                    "the smallest to the largest shot count"
                )
            mean_infidelity[(target.name, method)] = values

    nonphysical_fraction: dict[str, np.ndarray] = {}
    for target in targets:
        nonphysical_fraction[target.name] = np.asarray(
            [
                np.mean(
                    [
                        row["minimum_eigenvalue"] < -PHYSICAL_TOLERANCE
                        for row in rows_for(
                            records,
                            method="linear",
                            target=target.name,
                            shots=shots,
                        )
                    ]
                )
                for shots in SHOT_COUNTS
            ]
        )

    mle_rows = rows_for(records, method="mle")
    minimum_physical_eigenvalue = min(
        min(row["projected_minimum_eigenvalue"], row["mle_minimum_eigenvalue"])
        for row in mle_rows
    )
    minimum_nll_improvement = min(row["nll_improvement"] for row in mle_rows)
    if minimum_physical_eigenvalue < -PHYSICAL_TOLERANCE:
        raise AssertionError("a projected or MLE estimate was nonphysical")
    if minimum_nll_improvement < -2e-12:
        raise AssertionError("an MLE run increased its final objective")

    mean_iterations = np.asarray(
        [
            np.mean(
                [row["iterations"] for row in rows_for(records, method="mle", shots=shots)]
            )
            for shots in SHOT_COUNTS
        ]
    )
    mean_nll_improvement = np.asarray(
        [
            np.mean(
                [
                    row["nll_improvement"]
                    for row in rows_for(records, method="mle", shots=shots)
                ]
            )
            for shots in SHOT_COUNTS
        ]
    )

    return {
        **data,
        "rms_hs": rms_hs,
        "linear_slope": linear_slope,
        "mean_infidelity": mean_infidelity,
        "nonphysical_fraction": nonphysical_fraction,
        "minimum_physical_eigenvalue": minimum_physical_eigenvalue,
        "minimum_nll_improvement": minimum_nll_improvement,
        "mean_iterations": mean_iterations,
        "mean_nll_improvement": mean_nll_improvement,
    }


def print_report(results: dict[str, object]) -> None:
    """Print ensemble quality and optimizer diagnostics."""
    print(
        f"Finite-shot reconstruction verification: trials={results['trials']}, "
        f"seed={results['seed']}"
    )
    print(f"Linear HS-RMSE slope: {results['linear_slope']:.4f} (theory: -0.5)")
    print(
        f"Minimum projected/MLE eigenvalue: "
        f"{results['minimum_physical_eigenvalue']:+.3e}"
    )
    print(f"Minimum MLE NLL improvement: {results['minimum_nll_improvement']:+.3e}")

    print("\nEnsemble RMS Hilbert-Schmidt distance")
    print(f"{'shots':>7} {'linear':>12} {'projected':>12} {'MLE':>12}")
    print("-" * 48)
    for index, shots in enumerate(SHOT_COUNTS):
        print(
            f"{shots:7d} {results['rms_hs']['linear'][index]:12.4e} "
            f"{results['rms_hs']['projected'][index]:12.4e} "
            f"{results['rms_hs']['mle'][index]:12.4e}"
        )

    print("\nMean infidelity: first -> last shot count")
    for target in results["targets"]:
        for method in ("projected", "mle"):
            values = results["mean_infidelity"][(target.name, method)]
            print(f"  {target.name:<18} {method:<9} {values[0]:.4e} -> {values[-1]:.4e}")

    print("\nRaw linear-inversion nonphysical fraction")
    print(f"{'target':<18} " + " ".join(f"{shots:>8d}" for shots in SHOT_COUNTS))
    print("-" * 55)
    for target in results["targets"]:
        fractions = results["nonphysical_fraction"][target.name]
        print(
            f"{target.name:<18} "
            + " ".join(f"{fraction:8.2f}" for fraction in fractions)
        )
    print("\nAll finite-shot ensemble and physicality assertions passed.")


def plot_results(results: dict[str, object], output: Path, *, show: bool = False) -> None:
    """Plot accuracy scaling, physicality, and MLE optimizer behavior."""
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), constrained_layout=True)
    colors = {"linear": "#3569a8", "projected": "#cf6a32", "mle": "#3a8b5b"}
    markers = {"linear": "o", "projected": "s", "mle": "^"}

    scaling_axis = axes[0, 0]
    for method in ("linear", "projected", "mle"):
        scaling_axis.loglog(
            SHOT_COUNTS,
            results["rms_hs"][method],
            marker=markers[method],
            color=colors[method],
            label=method,
        )
    reference = results["rms_hs"]["linear"][0] * np.sqrt(
        SHOT_COUNTS[0] / np.asarray(SHOT_COUNTS)
    )
    scaling_axis.loglog(
        SHOT_COUNTS,
        reference,
        linestyle="--",
        color="0.35",
        label=r"$N^{-1/2}$ reference",
    )
    scaling_axis.set_xlabel("shots per Pauli setting")
    scaling_axis.set_ylabel("ensemble RMS HS distance")
    scaling_axis.set_title(f"Finite-shot scaling (LI slope={results['linear_slope']:.3f})")
    scaling_axis.grid(which="both", alpha=0.22)
    scaling_axis.legend()

    fidelity_axis = axes[0, 1]
    target_colors = ("#3569a8", "#9b59b6", "#cf6a32")
    for target, color in zip(results["targets"], target_colors):
        for method, linestyle, marker in (
            ("projected", "--", "s"),
            ("mle", "-", "o"),
        ):
            fidelity_axis.loglog(
                SHOT_COUNTS,
                np.maximum(results["mean_infidelity"][(target.name, method)], 1e-8),
                color=color,
                linestyle=linestyle,
                marker=marker,
                label=f"{target.name}, {method}",
            )
    fidelity_axis.set_xlabel("shots per Pauli setting")
    fidelity_axis.set_ylabel(r"mean physical-estimate infidelity $1-F$")
    fidelity_axis.set_title("Reconstruction quality by target")
    fidelity_axis.grid(which="both", alpha=0.22)
    fidelity_axis.legend(fontsize=8)

    physicality_axis = axes[1, 0]
    for target, color in zip(results["targets"], target_colors):
        physicality_axis.plot(
            SHOT_COUNTS,
            results["nonphysical_fraction"][target.name],
            marker="o",
            color=color,
            label=target.name,
        )
    physicality_axis.axhline(0.0, color="#3a8b5b", linewidth=2, label="projected/MLE: 0")
    physicality_axis.set_xscale("log")
    physicality_axis.set_ylim(-0.04, 1.04)
    physicality_axis.set_xlabel("shots per Pauli setting")
    physicality_axis.set_ylabel("nonphysical reconstruction fraction")
    physicality_axis.set_title("Raw linear inversion can leave state space")
    physicality_axis.grid(alpha=0.22)
    physicality_axis.legend(fontsize=8)

    optimizer_axis = axes[1, 1]
    history = results["representative_history"]
    optimizer_axis.plot(
        np.arange(len(history)),
        history,
        color="#3a8b5b",
        marker="o",
        markersize=3,
        label="Bell, 256 shots",
    )
    optimizer_axis.set_xlabel("accepted MLE iteration")
    optimizer_axis.set_ylabel("normalized negative log likelihood")
    optimizer_axis.set_title(
        "Representative monotone MLE run\n"
        f"NLL improvement={history[0] - history[-1]:.3e}"
    )
    optimizer_axis.grid(alpha=0.22)
    optimizer_axis.legend()

    figure.suptitle(
        f"Finite-shot reconstruction verification ({results['trials']} trials per condition)",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = analyze_trials(collect_trials(args.trials, args.seed))
    print_report(results)
    plot_results(results, args.output, show=args.show)
    print(f"Figure written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
