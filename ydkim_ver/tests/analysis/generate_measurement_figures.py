"""Generate figures for :mod:`measurement_generation.pauli_measurement`.

Run from the project root with::

    python tests/analysis/generate_measurement_figures.py

Every figure is saved as a 300-DPI PNG and a vector PDF under
``docs/figures/measurement``. Randomness comes only
from the stdlib-backed measurement functions and fixed integer seeds.

AI disclosure: this plotting script and its figure design were generated with
AI assistance on 2026-08-17. Results must be independently reviewed before the
figures are marked verified or included in the final submission.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nbqs_qst.measurement_generation import pauli_measurement as pm  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures" / "measurement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#5F6368",
}
AI_FOOTER = "AI-assisted figure generation · fixed seeds · independently verify"


def configure_style():
    """Apply a consistent, colorblind-safe poster style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, stem):
    """Save one poster figure in raster and vector formats."""
    fig.text(0.995, 0.006, AI_FOOTER, ha="right", va="bottom", fontsize=6)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(png_path), str(pdf_path)


def density_matrix(state):
    """Pure-state density matrix."""
    state = np.asarray(state, dtype=np.complex128)
    return np.outer(state, state.conj())


def ghz_state(n):
    """Return the n-qubit GHZ vector."""
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 2**-0.5
    state[-1] = 2**-0.5
    return state


def deterministic_random_state(n, seed):
    """Complex normalized state generated only from stdlib ``Random``."""
    rng = random.Random(seed)
    state = np.asarray(
        [
            rng.gauss(0.0, 1.0) + 1j * rng.gauss(0.0, 1.0)
            for _ in range(2**n)
        ],
        dtype=np.complex128,
    )
    return state / np.linalg.norm(state)


def parity_expectation(outcomes, n, support):
    """Mean product of eigenvalues on the selected qubits."""
    outcomes = np.asarray(outcomes)
    parity = np.ones(outcomes.shape[0], dtype=np.int64)
    for qubit in support:
        parity *= 1 - 2 * ((outcomes >> (n - 1 - qubit)) & 1)
    return float(np.mean(parity))


def plot_shot_scaling(results):
    """Show the finite-sampling 1/sqrt(N) law."""
    rho = density_matrix([1, 0, 0, 0])
    shot_counts = np.asarray([32, 64, 128, 256, 512, 1024, 2048, 4096])
    trials = 120
    rms = []
    q16 = []
    q84 = []

    for shots in shot_counts:
        estimates = []
        for trial in range(trials):
            outcomes = pm.sample_outcomes(
                rho, "XX", int(shots), seed=10_000 + trial
            )
            estimates.append(parity_expectation(outcomes, 2, (0, 1)))
        errors = np.asarray(estimates)  # exact <XX> for |00> is zero
        rms.append(float(np.sqrt(np.mean(errors**2))))
        q16.append(float(np.quantile(np.abs(errors), 0.16)))
        q84.append(float(np.quantile(np.abs(errors), 0.84)))

    rms = np.asarray(rms)
    slope, intercept = np.polyfit(np.log(shot_counts), np.log(rms), 1)
    reference = np.exp(intercept) * shot_counts**-0.5

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.loglog(
        shot_counts,
        rms,
        "o-",
        color=COLORS["blue"],
        label="Monte Carlo RMS error (120 trials)",
    )
    ax.loglog(
        shot_counts,
        reference,
        "--",
        color=COLORS["gray"],
        label=r"Reference $N^{-1/2}$",
    )
    ax.set_title(r"Finite-shot noise follows the expected $1/\sqrt{N}$ law")
    ax.set_xlabel("Shots per setting, N")
    ax.set_ylabel(r"RMS error of $\widehat{\langle XX\rangle}$")
    ax.legend(frameon=False)
    ax.text(
        0.05,
        0.08,
        f"fitted log–log slope = {slope:.3f}",
        transform=ax.transAxes,
        color=COLORS["blue"],
    )
    paths = save_figure(fig, "poster_shot_scaling")
    results["shot_scaling"] = {
        "shots": shot_counts.tolist(),
        "rms_error": rms.tolist(),
        "absolute_error_q16": q16,
        "absolute_error_q84": q84,
        "fitted_slope": float(slope),
        "files": paths,
    }


def _plot_expectation_clouds(
    results,
    expectation_values,
    series_colors,
    *,
    stem,
    result_key,
    seed_base,
):
    """Create one finite-shot scatter plot for selected true values."""
    shot_counts = np.asarray([10, 30, 100, 300, 1000, 3000, 10_000])
    trials = 80
    rms_by_value = {}

    fig, ax = plt.subplots(figsize=(8.1, 5.0))
    for value_index, (true_value, color) in enumerate(
        zip(expectation_values, series_colors)
    ):
        rho = np.asarray(
            [
                [(1 + true_value) / 2, 0],
                [0, (1 - true_value) / 2],
            ],
            dtype=np.complex128,
        )
        rms = []
        for shot_index, shots in enumerate(shot_counts):
            estimates = []
            for trial in range(trials):
                seed = (
                    seed_base
                    + value_index * 10_000
                    + shot_index * 1000
                    + trial
                )
                outcomes = pm.sample_outcomes(
                    rho, "Z", int(shots), seed=seed
                )
                estimates.append(float(np.mean(1 - 2 * np.asarray(outcomes))))
            estimates = np.asarray(estimates)
            rms.append(float(np.sqrt(np.mean((estimates - true_value) ** 2))))
            ax.scatter(
                np.full(trials, shots),
                estimates,
                s=13,
                alpha=0.13,
                color=color,
                linewidths=0,
            )
        rms_by_value[f"{true_value:+.1f}"] = rms
        ax.plot(
            shot_counts,
            np.full(shot_counts.shape, true_value, dtype=float),
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=4,
        )
        ax.text(
            11_500,
            true_value,
            rf"$\langle Z\rangle={true_value:g}$",
            color=color,
            va="center",
            fontsize=10,
        )

    ax.set_xscale("log")
    ax.set_xlim(7.5, 20_000)
    ax.set_ylim(-1.08, 1.08)
    ax.set_title("Finite-shot estimates concentrate around the true expectation")
    ax.set_xlabel("Shots per setting, N")
    ax.set_ylabel(r"Estimated $\widehat{\langle Z\rangle}$")
    ax.text(
        0.04,
        0.08,
        r"$\mathrm{Var}(\widehat{\langle Z\rangle})=(1-\langle Z\rangle^2)/N$",
        transform=ax.transAxes,
        color=COLORS["gray"],
    )
    paths = save_figure(fig, stem)
    results[result_key] = {
        "true_expectations": list(expectation_values),
        "shots": shot_counts.tolist(),
        "trials": trials,
        "rms_error_by_expectation": rms_by_value,
        "files": paths,
    }


def plot_five_expectation_clouds(results):
    """Show scatter for true values 1, 0.5, 0, -0.5, and -1."""
    _plot_expectation_clouds(
        results,
        (1.0, 0.5, 0.0, -0.5, -1.0),
        (
            COLORS["green"],
            COLORS["orange"],
            COLORS["blue"],
            COLORS["purple"],
            COLORS["red"],
        ),
        stem="poster_five_expectation_shot_clouds",
        result_key="five_expectation_shot_clouds",
        seed_base=50_000,
    )


def plot_three_expectation_clouds(results):
    """Show scatter for the compact reference set 1, 0, and -1."""
    _plot_expectation_clouds(
        results,
        (1.0, 0.0, -1.0),
        (COLORS["green"], COLORS["blue"], COLORS["red"]),
        stem="poster_three_expectation_shot_clouds",
        result_key="three_expectation_shot_clouds",
        seed_base=90_000,
    )


def plot_aggregation_gain(results):
    """Demonstrate the benefit of pooling compatible settings."""
    plus_y = np.asarray([1, 1j], dtype=np.complex128) / np.sqrt(2)
    zero = np.asarray([1, 0], dtype=np.complex128)
    rho = density_matrix(np.kron(plus_y, zero))
    shots = 128
    trials = 240
    last_only = []
    pooled = []

    for trial in range(trials):
        dataset = pm.generate_measurement_dataset(
            rho, shots, seed=20_000 + trial
        )
        pooled.append(pm.expectations_from_dataset(dataset)["XI"])
        xz_index = dataset.settings.index("XZ")
        last_only.append(
            parity_expectation(dataset.outcomes[xz_index], 2, (0,))
        )

    last_only = np.asarray(last_only)
    pooled = np.asarray(pooled)
    rms_last = float(np.sqrt(np.mean(last_only**2)))
    rms_pooled = float(np.sqrt(np.mean(pooled**2)))
    bins = np.linspace(-0.34, 0.34, 28)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(
        last_only,
        bins=bins,
        density=True,
        alpha=0.48,
        color=COLORS["orange"],
        label=f"One compatible setting (N={shots}), RMS={rms_last:.3f}",
    )
    ax.hist(
        pooled,
        bins=bins,
        density=True,
        alpha=0.58,
        color=COLORS["blue"],
        label=f"All 3 settings (3N={3 * shots}), RMS={rms_pooled:.3f}",
    )
    ax.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=1.4)
    ax.set_title("Pooling compatible settings narrows the estimator")
    ax.set_xlabel(r"Estimated $\langle XI\rangle$ (true value 0)")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)
    ax.text(
        0.04,
        0.92,
        f"observed RMS ratio = {rms_pooled / rms_last:.3f}\n"
        f"ideal ratio = 1/√3 = {1 / np.sqrt(3):.3f}",
        transform=ax.transAxes,
        va="top",
    )
    paths = save_figure(fig, "poster_aggregation_gain")
    results["aggregation_gain"] = {
        "shots_per_setting": shots,
        "trials": trials,
        "rms_one_setting": rms_last,
        "rms_all_compatible_settings": rms_pooled,
        "rms_ratio": rms_pooled / rms_last,
        "files": paths,
    }


def benchmark_exact(rho, repeats=15):
    """Median wall time of the exact expectation path."""
    pm.pauli_expectations(rho)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        pm.pauli_expectations(rho)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def plot_backend_comparison(results):
    """Compare NumPy and JAX accuracy, seeded sampling, and runtime."""
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("JAX is required for the backend poster plot") from exc

    rho3 = density_matrix(deterministic_random_state(3, seed=61))
    exact_numpy = pm.pauli_expectations(rho3)
    exact_jax = pm.pauli_expectations(jnp.asarray(rho3))
    keys = list(exact_numpy)
    values_numpy = np.asarray([exact_numpy[key] for key in keys])
    values_jax = np.asarray([exact_jax[key] for key in keys])
    exact_delta = float(np.max(np.abs(values_numpy - values_jax)))

    sampled_numpy = pm.pauli_expectations(rho3, shots=128, seed=31)
    sampled_jax = pm.pauli_expectations(jnp.asarray(rho3), shots=128, seed=31)
    sampled_delta = max(
        abs(sampled_numpy[key] - sampled_jax[key]) for key in keys
    )

    qubits = np.arange(1, 6)
    numpy_times = []
    jax_times = []
    for n in qubits:
        rho = density_matrix(ghz_state(int(n)))
        numpy_times.append(benchmark_exact(rho))
        jax_times.append(benchmark_exact(jnp.asarray(rho)))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axes[0]
    ax.scatter(
        values_numpy,
        values_jax,
        s=34,
        alpha=0.75,
        color=COLORS["blue"],
        edgecolors="none",
    )
    ax.plot([-1.05, 1.05], [-1.05, 1.05], "--", color=COLORS["gray"])
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Exact 3-qubit expectations")
    ax.set_xlabel("NumPy")
    ax.set_ylabel("JAX")
    ax.text(
        0.04,
        0.94,
        f"max exact |Δ| = {exact_delta:.1e}\n"
        f"fixed-seed sampled |Δ| = {sampled_delta:.1e}",
        transform=ax.transAxes,
        va="top",
    )

    ax = axes[1]
    ax.semilogy(
        qubits,
        np.asarray(numpy_times) * 1e3,
        "o-",
        color=COLORS["blue"],
        label="NumPy (CPU)",
    )
    ax.semilogy(
        qubits,
        np.asarray(jax_times) * 1e3,
        "s-",
        color=COLORS["red"],
        label="JAX (CPU, eager)",
    )
    ax.set_title("Exact-path runtime")
    ax.set_xlabel("Qubits, n")
    ax.set_ylabel("Median wall time (ms)")
    ax.set_xticks(qubits)
    ax.legend(frameon=False)
    fig.suptitle("Backend portability: identical results from the same code")
    paths = save_figure(fig, "poster_backend_comparison")
    results["backend_comparison"] = {
        "qubits": qubits.tolist(),
        "numpy_time_seconds": numpy_times,
        "jax_time_seconds": jax_times,
        "max_exact_difference": exact_delta,
        "max_seeded_sample_difference": sampled_delta,
        "jax_platform": jax.default_backend(),
        "files": paths,
    }


def plot_bell_measurements(results):
    """Compare sampled Bell-state outcome frequencies with Born probabilities."""
    rho = density_matrix(ghz_state(2))
    shots = 1000
    dataset = pm.generate_measurement_dataset(rho, shots, seed=47)
    settings = ("XX", "YY", "ZZ")
    theory = {
        "XX": np.asarray([0.5, 0.0, 0.0, 0.5]),
        "YY": np.asarray([0.0, 0.5, 0.5, 0.0]),
        "ZZ": np.asarray([0.5, 0.0, 0.0, 0.5]),
    }
    outcomes = ("00", "01", "10", "11")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1), sharey=True)
    sampled_values = {}
    x = np.arange(4)
    width = 0.36
    for ax, basis in zip(axes, settings):
        index = dataset.settings.index(basis)
        sampled = np.asarray(dataset.counts[index], dtype=float) / shots
        sampled_values[basis] = sampled.tolist()
        ax.bar(
            x - width / 2,
            theory[basis],
            width,
            color=COLORS["gray"],
            alpha=0.7,
            label="Born probability",
        )
        ax.bar(
            x + width / 2,
            sampled,
            width,
            color=COLORS["blue"],
            alpha=0.88,
            label="Sampled frequency",
        )
        ax.set_title(f"{basis} basis")
        ax.set_xticks(x, outcomes)
        ax.set_xlabel("Outcome")
        ax.set_ylim(0, 0.58)
    axes[0].set_ylabel("Probability / frequency")
    axes[0].legend(frameon=False, loc="upper center")
    fig.suptitle(r"Bell-state measurements reproduce $p_x=\mathrm{Tr}(E_x\rho)$")
    paths = save_figure(fig, "poster_bell_measurements")
    results["bell_measurements"] = {
        "shots_per_setting": shots,
        "sampled_frequencies": sampled_values,
        "files": paths,
    }


def plot_pauli_validation(results):
    """Show exact values for reference states, including the odd-Y sign."""
    plus_y = np.asarray([1, 1j], dtype=np.complex128) / np.sqrt(2)
    states = {
        r"$|00\rangle$": (
            density_matrix([1, 0, 0, 0]),
            {"II": 1, "ZI": 1, "IZ": 1, "ZZ": 1},
        ),
        "Bell": (
            density_matrix(ghz_state(2)),
            {"II": 1, "XX": 1, "YY": -1, "ZZ": 1},
        ),
        r"$|+y\rangle$": (
            density_matrix(plus_y),
            {"I": 1, "X": 0, "Y": 1, "Z": 0},
        ),
        r"GHZ$_3$": (
            density_matrix(ghz_state(3)),
            {"III": 1, "XXX": 1, "ZZI": 1, "IZZ": 1},
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    errors = {}
    for ax, (name, (rho, expected)) in zip(axes.flat, states.items()):
        measured = pm.pauli_expectations(rho)
        labels = list(expected)
        theory_values = np.asarray([expected[label] for label in labels])
        measured_values = np.asarray([measured[label] for label in labels])
        errors[name] = float(np.max(np.abs(theory_values - measured_values)))
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(
            x - width / 2,
            theory_values,
            width,
            color=COLORS["gray"],
            alpha=0.72,
            label="Theory",
        )
        ax.bar(
            x + width / 2,
            measured_values,
            width,
            color=COLORS["orange"],
            alpha=0.9,
            label="Implementation",
        )
        ax.set_title(f"{name}   max |Δ|={errors[name]:.1e}")
        ax.set_xticks(x, labels)
        ax.set_ylim(-1.18, 1.18)
        ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    axes[0, 0].set_ylabel("Pauli expectation")
    axes[1, 0].set_ylabel("Pauli expectation")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("Reference-state regression, including the Y-sign convention")
    paths = save_figure(fig, "poster_pauli_validation")
    results["pauli_validation"] = {"max_errors": errors, "files": paths}


def plot_resource_scaling(results):
    """Make the exhaustive tomography scaling limit explicit."""
    qubits = np.arange(1, 9)
    settings = 3.0**qubits
    shots_per_setting = 1000
    total_shots = shots_per_setting * settings
    pauli_memory_gib = (16.0**qubits * 16) / 2**30

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax = axes[0]
    ax.semilogy(qubits, settings, "o-", color=COLORS["green"])
    ax.set_title("Measurement settings")
    ax.set_xlabel("Qubits, n")
    ax.set_ylabel(r"Settings, $3^n$")
    ax.set_xticks(qubits)
    ax.text(
        0.05,
        0.9,
        f"At N={shots_per_setting:,}: total shots = N·3ⁿ",
        transform=ax.transAxes,
        va="top",
    )

    ax = axes[1]
    ax.semilogy(qubits, pauli_memory_gib, "s-", color=COLORS["purple"])
    ax.axhline(1, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    ax.set_title("Exact-path Pauli tensor")
    ax.set_xlabel("Qubits, n")
    ax.set_ylabel("Theoretical array size (GiB)")
    ax.set_xticks(qubits)
    ax.text(
        0.05,
        0.9,
        r"complex128 storage: $16^n\times16$ bytes",
        transform=ax.transAxes,
        va="top",
    )
    fig.suptitle("Exhaustive Pauli tomography is exponentially costly")
    paths = save_figure(fig, "poster_resource_scaling")
    results["resource_scaling"] = {
        "qubits": qubits.tolist(),
        "settings": settings.astype(int).tolist(),
        "shots_per_setting": shots_per_setting,
        "total_shots": total_shots.astype(int).tolist(),
        "pauli_array_gib": pauli_memory_gib.tolist(),
        "files": paths,
    }


def main():
    configure_style()
    results = {
        "metadata": {
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "ai_disclosure": AI_FOOTER,
        }
    }
    plot_shot_scaling(results)
    plot_five_expectation_clouds(results)
    plot_three_expectation_clouds(results)
    plot_aggregation_gain(results)
    plot_backend_comparison(results)
    plot_bell_measurements(results)
    plot_pauli_validation(results)
    plot_resource_scaling(results)
    with (OUTPUT_DIR / "poster_plot_data.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"Generated {8} figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
