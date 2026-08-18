"""Verify finite-shot Pauli sampling against multinomial statistics.

Run from the repository root with::

    python verification/measurement_simulation/sampling_distribution_verification.py

The script repeats one nontrivial two-qubit measurement over independent draws
and checks its mean, covariance, and inverse-square-root shot scaling.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nbqst.measurements import simulate_pauli_measurements  # noqa: E402

from born_probability_verification import projector_oracle  # noqa: E402


SETTING = "XY"
SHOT_COUNTS = (32, 128, 512, 2048)
EXPECTATION_SHOT_COUNTS = (100, 300, 1000, 3000, 10000)
DEFAULT_TRIALS = 1200
DEFAULT_CLOUD_TRIALS = 160
DEFAULT_SEED = 20260819
DEFAULT_OUTPUT = Path(__file__).with_name("finite_shot_sampling_verification.png")
DEFAULT_CONCENTRATION_OUTPUT = Path(__file__).with_name(
    "pauli_expectation_concentration.png"
)


@dataclass(frozen=True)
class Check:
    """One scalar acceptance check shown in the terminal report."""

    name: str
    observed: float
    limit: float
    context: str

    @property
    def passed(self) -> bool:
        return self.observed <= self.limit


def validation_state() -> np.ndarray:
    """Return a fixed full-rank state with four nontrivial XY probabilities."""
    generator = np.random.default_rng(81357)
    matrix = generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


def collect_trials(trials: int, seed: int) -> dict[str, object]:
    """Collect frequency vectors using the public measurement simulator."""
    if trials < 100:
        raise ValueError("trials must be at least 100 for stable covariance checks")

    rho = validation_state()
    probabilities = projector_oracle(rho, SETTING)
    generator = np.random.default_rng(seed)
    frequencies: dict[int, np.ndarray] = {}
    counts_by_shots: dict[int, np.ndarray] = {}

    for shots in SHOT_COUNTS:
        counts = np.empty((trials, 4), dtype=np.int64)
        for trial in range(trials):
            dataset = simulate_pauli_measurements(
                rho,
                shots,
                settings=(SETTING,),
                rng=generator,
            )
            counts[trial] = np.asarray(dataset.counts[SETTING], dtype=np.int64)
        if not np.all(counts.sum(axis=1) == shots):
            raise AssertionError(f"N={shots}: multinomial counts do not conserve shots")
        if np.any(counts < 0):
            raise AssertionError(f"N={shots}: a sampled count is negative")
        counts_by_shots[shots] = counts
        frequencies[shots] = counts / shots

    return {
        "rho": rho,
        "probabilities": probabilities,
        "counts": counts_by_shots,
        "frequencies": frequencies,
    }


def analyze_trials(data: dict[str, object], trials: int) -> dict[str, object]:
    """Calculate empirical and theoretical multinomial diagnostics."""
    probabilities = np.asarray(data["probabilities"])
    target_scaled_covariance = np.diag(probabilities) - np.outer(
        probabilities, probabilities
    )
    means = {}
    variances = {}
    scaled_covariances = {}
    rms_errors = []
    mean_z_scores = []
    covariance_z_scores = []

    covariance_standard_error = np.sqrt(
        (
            target_scaled_covariance**2
            + np.outer(np.diag(target_scaled_covariance), np.diag(target_scaled_covariance))
        )
        / (trials - 1)
    )

    for shots in SHOT_COUNTS:
        frequency = data["frequencies"][shots]
        mean = np.mean(frequency, axis=0)
        covariance = np.cov(frequency, rowvar=False, ddof=1)
        scaled_covariance = shots * covariance
        variance = np.diag(covariance)
        standard_error = np.sqrt(probabilities * (1.0 - probabilities) / (shots * trials))

        means[shots] = mean
        variances[shots] = variance
        scaled_covariances[shots] = scaled_covariance
        rms_errors.append(float(np.sqrt(np.mean((frequency - probabilities) ** 2))))
        mean_z_scores.extend(np.abs(mean - probabilities) / standard_error)
        covariance_z_scores.extend(
            np.ravel(
                np.abs(scaled_covariance - target_scaled_covariance)
                / covariance_standard_error
            )
        )

    rms_errors_array = np.asarray(rms_errors)
    theoretical_rms = np.asarray(
        [
            math.sqrt(float(np.mean(probabilities * (1.0 - probabilities))) / shots)
            for shots in SHOT_COUNTS
        ]
    )
    slope = float(np.polyfit(np.log(SHOT_COUNTS), np.log(rms_errors_array), 1)[0])

    checks = (
        Check(
            "Maximum mean z-score",
            float(np.max(mean_z_scores)),
            6.0,
            "all outcomes and shot counts",
        ),
        Check(
            "Maximum covariance z-score",
            float(np.max(covariance_z_scores)),
            7.0,
            "Gaussian covariance-SE approximation",
        ),
        Check(
            "RMSE slope error",
            abs(slope + 0.5),
            0.10,
            f"fitted slope={slope:.4f}, theory=-0.5",
        ),
        Check(
            "RMSE/theory relative error",
            float(np.max(np.abs(rms_errors_array / theoretical_rms - 1.0))),
            0.08,
            "maximum over shot counts",
        ),
    )

    return {
        **data,
        "target_scaled_covariance": target_scaled_covariance,
        "means": means,
        "variances": variances,
        "scaled_covariances": scaled_covariances,
        "rms_errors": rms_errors_array,
        "theoretical_rms": theoretical_rms,
        "slope": slope,
        "checks": checks,
    }


def collect_expectation_clouds(trials: int, seed: int) -> dict[float, dict[int, np.ndarray]]:
    """Sample Z-expectation estimates for true values +1, 0, and -1."""
    if trials < 20:
        raise ValueError("cloud trials must be at least 20")

    zero = np.asarray([1, 0], dtype=np.complex128)
    one = np.asarray([0, 1], dtype=np.complex128)
    plus = np.asarray([1, 1], dtype=np.complex128) / np.sqrt(2.0)
    states = {
        1.0: np.outer(zero, zero.conj()),
        0.0: np.outer(plus, plus.conj()),
        -1.0: np.outer(one, one.conj()),
    }
    generator = np.random.default_rng(seed)
    clouds: dict[float, dict[int, np.ndarray]] = {}

    for expectation, rho in states.items():
        by_shots: dict[int, np.ndarray] = {}
        for shots in EXPECTATION_SHOT_COUNTS:
            estimates = np.empty(trials)
            for trial in range(trials):
                dataset = simulate_pauli_measurements(
                    rho,
                    shots,
                    settings=("Z",),
                    rng=generator,
                )
                counts = np.asarray(dataset.counts["Z"], dtype=np.int64)
                estimates[trial] = (counts[0] - counts[1]) / shots
            by_shots[shots] = estimates
        clouds[expectation] = by_shots

    for expectation in (-1.0, 1.0):
        if any(
            not np.allclose(values, expectation, atol=0.0, rtol=0.0)
            for values in clouds[expectation].values()
        ):
            raise AssertionError("a Pauli eigenstate produced a fluctuating expectation")

    for shots, estimates in clouds[0.0].items():
        mean_standard_error = 1.0 / math.sqrt(shots * trials)
        if abs(float(np.mean(estimates))) > 6.0 * mean_standard_error:
            raise AssertionError(f"N={shots}: zero-expectation cloud has a biased mean")

    return clouds


def print_report(results: dict[str, object], trials: int) -> None:
    """Print probabilities, empirical moments, and acceptance checks."""
    probabilities = results["probabilities"]
    print(f"Finite-shot verification: setting={SETTING}, trials={trials}")
    print("Independent projector-oracle probabilities:")
    print("  " + ", ".join(f"p({outcome:02b})={value:.6f}" for outcome, value in enumerate(probabilities)))
    print(f"\n{'shots':>7} {'max mean error':>16} {'RMSE':>12} {'theory':>12}")
    print("-" * 53)
    for index, shots in enumerate(SHOT_COUNTS):
        mean_error = np.max(np.abs(results["means"][shots] - probabilities))
        print(
            f"{shots:7d} {mean_error:16.4e} {results['rms_errors'][index]:12.4e} "
            f"{results['theoretical_rms'][index]:12.4e}"
        )

    print(f"\n{'check':<34} {'observed':>11} {'limit':>11} {'result':>8}")
    print("-" * 70)
    for check in results["checks"]:
        result = "PASS" if check.passed else "FAIL"
        print(f"{check.name:<34} {check.observed:11.4e} {check.limit:11.4e} {result:>8}")
        print(f"  {check.context}")


def plot_results(
    results: dict[str, object], trials: int, output: Path, *, show: bool = False
) -> None:
    """Plot means, variances, RMSE scaling, and covariance agreement."""
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=True)
    probabilities = results["probabilities"]
    outcome_indices = np.arange(4)
    outcome_labels = ["00", "01", "10", "11"]

    display_shots = 128
    empirical_mean = results["means"][display_shots]
    mean_se = np.sqrt(
        probabilities * (1.0 - probabilities) / (display_shots * trials)
    )
    width = 0.36
    axes[0, 0].bar(
        outcome_indices - width / 2,
        probabilities,
        width,
        label="Born probability",
        color="#3569a8",
    )
    axes[0, 0].bar(
        outcome_indices + width / 2,
        empirical_mean,
        width,
        yerr=3.0 * mean_se,
        capsize=4,
        label=f"empirical mean ({trials} trials)",
        color="#cf6a32",
    )
    axes[0, 0].set_xticks(outcome_indices, outcome_labels)
    axes[0, 0].set_xlabel("XY outcome bit string")
    axes[0, 0].set_ylabel("probability / mean frequency")
    axes[0, 0].set_title(f"A. Mean frequencies at N={display_shots}")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.22)

    colors = ("#3569a8", "#cf6a32", "#3a8f63", "#8f5da2")
    for outcome, color in enumerate(colors):
        empirical = np.asarray(
            [results["variances"][shots][outcome] for shots in SHOT_COUNTS]
        )
        theory = probabilities[outcome] * (1.0 - probabilities[outcome]) / np.asarray(
            SHOT_COUNTS
        )
        axes[0, 1].loglog(
            SHOT_COUNTS,
            empirical,
            "o-",
            color=color,
            label=f"{outcome:02b} empirical",
        )
        axes[0, 1].loglog(SHOT_COUNTS, theory, "--", color=color, alpha=0.75)
    axes[0, 1].set_xlabel("shots per setting N")
    axes[0, 1].set_ylabel(r"frequency variance $\mathrm{Var}(\hat p_b)$")
    axes[0, 1].set_title("B. Binomial marginal variance")
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[0, 1].grid(which="both", alpha=0.22)

    axes[1, 0].loglog(
        SHOT_COUNTS,
        results["rms_errors"],
        "o-",
        color="#3569a8",
        linewidth=2,
        label=f"empirical, slope={results['slope']:.3f}",
    )
    axes[1, 0].loglog(
        SHOT_COUNTS,
        results["theoretical_rms"],
        "--",
        color="#cf6a32",
        linewidth=2,
        label=r"multinomial theory, $N^{-1/2}$",
    )
    axes[1, 0].set_xlabel("shots per setting N")
    axes[1, 0].set_ylabel("frequency-vector RMSE")
    axes[1, 0].set_title("C. Finite-shot error scaling")
    axes[1, 0].legend()
    axes[1, 0].grid(which="both", alpha=0.22)

    target = results["target_scaled_covariance"]
    all_target = []
    all_empirical = []
    for shots in SHOT_COUNTS:
        all_target.extend(target.ravel())
        all_empirical.extend(results["scaled_covariances"][shots].ravel())
    all_target = np.asarray(all_target)
    all_empirical = np.asarray(all_empirical)
    minimum = min(all_target.min(), all_empirical.min())
    maximum = max(all_target.max(), all_empirical.max())
    padding = 0.06 * (maximum - minimum)
    limits = [minimum - padding, maximum + padding]
    axes[1, 1].scatter(all_target, all_empirical, s=28, alpha=0.6, color="#3a8f63")
    axes[1, 1].plot(limits, limits, color="0.25", linestyle="--", linewidth=1.2)
    axes[1, 1].set_xlim(limits)
    axes[1, 1].set_ylim(limits)
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].set_xlabel(r"theory $[\mathrm{diag}(p)-pp^T]_{ij}$")
    axes[1, 1].set_ylabel(r"empirical $N\,\mathrm{Cov}(\hat p)_{ij}$")
    axes[1, 1].set_title("D. Full multinomial covariance")
    axes[1, 1].grid(alpha=0.22)

    figure.suptitle("Finite-shot local-Pauli sampling verification", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_expectation_concentration(
    clouds: dict[float, dict[int, np.ndarray]],
    output: Path,
    *,
    show: bool = False,
) -> None:
    """Show finite-shot Pauli estimates concentrating around their true values."""
    figure, axis = plt.subplots(figsize=(13.5, 7.7), constrained_layout=True)
    shot_array = np.asarray(EXPECTATION_SHOT_COUNTS, dtype=float)

    for shots in EXPECTATION_SHOT_COUNTS:
        values = clouds[0.0][shots]
        axis.scatter(
            np.full(values.size, shots),
            values,
            s=30,
            alpha=0.18,
            color="#167bb5",
            edgecolors="none",
            zorder=3,
        )

    standard_deviation = 1.0 / np.sqrt(shot_array)
    axis.fill_between(
        shot_array,
        -2.0 * standard_deviation,
        2.0 * standard_deviation,
        color="#167bb5",
        alpha=0.12,
        zorder=1,
    )
    axis.plot(
        shot_array,
        np.zeros_like(shot_array),
        "o-",
        color="#167bb5",
        linewidth=2.2,
        markersize=7,
        zorder=4,
    )
    axis.plot(
        shot_array,
        np.ones_like(shot_array),
        "o-",
        color="#0b9f77",
        linewidth=2.2,
        markersize=7,
        zorder=4,
    )
    axis.plot(
        shot_array,
        -np.ones_like(shot_array),
        "o-",
        color="#d65f00",
        linewidth=2.2,
        markersize=7,
        zorder=4,
    )

    axis.set_xscale("log")
    axis.set_xlim(80, 15000)
    axis.set_ylim(-1.1, 1.1)
    axis.set_xticks(EXPECTATION_SHOT_COUNTS)
    axis.set_xticklabels([f"{shots:,}" for shots in EXPECTATION_SHOT_COUNTS])
    axis.set_xlabel("shots per setting, N")
    axis.set_ylabel(r"estimated Pauli expectation $\widehat{\langle Z\rangle}$")
    axis.set_title("Finite-shot Pauli estimates concentrate around the true expectation")
    axis.grid(alpha=0.24)
    axis.text(
        330,
        0.14,
        r"theoretical $\pm2\sigma$ band",
        color="#167bb5",
        fontsize=11,
    )
    axis.text(
        105,
        -0.84,
        r"$\mathrm{Var}(\widehat{\langle Z\rangle})"
        r"=(1-\langle Z\rangle^2)/N$",
        color="0.35",
        fontsize=13,
    )
    axis.annotate(
        r"$\langle Z\rangle=+1$",
        xy=(10000, 1.0),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
        color="#0b9f77",
        fontsize=12,
    )
    axis.annotate(
        r"$\langle Z\rangle=0$",
        xy=(10000, 0.0),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
        color="#167bb5",
        fontsize=12,
    )
    axis.annotate(
        r"$\langle Z\rangle=-1$",
        xy=(10000, -1.0),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
        color="#d65f00",
        fontsize=12,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--cloud-trials", type=int, default=DEFAULT_CLOUD_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--concentration-output",
        type=Path,
        default=DEFAULT_CONCENTRATION_OUTPUT,
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = collect_trials(args.trials, args.seed)
    results = analyze_trials(data, args.trials)
    print_report(results, args.trials)
    plot_results(results, args.trials, args.output, show=args.show)
    clouds = collect_expectation_clouds(args.cloud_trials, args.seed + 1)
    plot_expectation_concentration(
        clouds,
        args.concentration_output,
        show=args.show,
    )
    print(f"\nFigure written to {args.output.resolve()}")
    print(f"Figure written to {args.concentration_output.resolve()}")

    failures = [check.name for check in results["checks"] if not check.passed]
    if failures:
        raise SystemExit("Finite-shot checks failed: " + ", ".join(failures))
    print("All finite-shot multinomial checks passed.")


if __name__ == "__main__":
    main()
