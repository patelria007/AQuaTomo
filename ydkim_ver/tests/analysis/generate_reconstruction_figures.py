"""Generate reproducible poster figures for state reconstruction.

Run from the project root with::

    python tests/analysis/generate_reconstruction_figures.py

Outputs are 300-DPI PNG, vector PDF, and exact JSON data.  Quantum sampling
uses only the stdlib-backed measurement module with fixed seeds.

AI disclosure: this plotting code and figure design were generated with AI
assistance on 2026-08-17. Independently review all results before use.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nbqs_qst.measurement_generation import generate_measurement_dataset  # noqa: E402
from nbqs_qst.state_reconstruction import (  # noqa: E402
    linear_inversion,
    maximum_likelihood,
    projected_least_squares,
    state_fidelity,
)

OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures" / "state_reconstruction"
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
    fig.text(0.995, 0.006, AI_FOOTER, ha="right", va="bottom", fontsize=6)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(png), str(pdf)


def density_matrix(state):
    state = np.asarray(state, dtype=np.complex128)
    return np.outer(state, state.conj())


def reference_states():
    plus_y = np.asarray([1, 1j], dtype=np.complex128) / np.sqrt(2)
    bell = np.asarray([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    ghz4 = np.zeros(16, dtype=np.complex128)
    ghz4[0] = 2**-0.5
    ghz4[-1] = 2**-0.5
    return {
        "plus_y": density_matrix(plus_y),
        "bell": density_matrix(bell),
        "ghz4": density_matrix(ghz4),
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "q16": float(np.quantile(values, 0.16)),
        "q84": float(np.quantile(values, 0.84)),
    }


def plot_reconstruction_quality(results):
    """Compare PLS/MLE fidelity and expose nonphysical linear inversion."""
    states = reference_states()
    shots_values = np.asarray([32, 128, 512, 2048])
    trials = 20
    records = {}

    for state_index, (name, target) in enumerate(states.items()):
        records[name] = {"pls": [], "mle": [], "linear_nonphysical": []}
        for shot_index, shots in enumerate(shots_values):
            pls_infidelities = []
            mle_infidelities = []
            nonphysical = []
            mle_converged = []
            for trial in range(trials):
                seed = 10_000 * state_index + 1000 * shot_index + trial
                dataset = generate_measurement_dataset(
                    target, int(shots), seed=seed
                )
                linear = linear_inversion(dataset)
                pls = projected_least_squares(dataset)
                mle = maximum_likelihood(
                    dataset, max_iterations=400, tolerance=2e-8
                )
                nonphysical.append(
                    float(np.min(np.linalg.eigvalsh(linear)) < -1e-12)
                )
                pls_infidelities.append(
                    max(1e-12, 1.0 - state_fidelity(target, pls))
                )
                mle_infidelities.append(
                    max(1e-12, 1.0 - state_fidelity(target, mle.rho))
                )
                mle_converged.append(mle.converged)
            records[name]["pls"].append(summarize(pls_infidelities))
            records[name]["mle"].append(summarize(mle_infidelities))
            records[name]["linear_nonphysical"].append(
                float(np.mean(nonphysical))
            )
            records[name].setdefault("mle_converged", []).append(
                int(sum(mle_converged))
            )

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.5))
    titles = {
        "plus_y": r"One qubit: $|+y\rangle$",
        "bell": r"Two qubits: $|\Phi^+\rangle$",
        "ghz4": r"Four qubits: $|GHZ_4\rangle$",
    }
    for ax, name in zip(axes[:3], ("plus_y", "bell", "ghz4")):
        for method, color, marker, label in (
            ("pls", COLORS["orange"], "s", "Projected least squares"),
            ("mle", COLORS["blue"], "o", "Multinomial MLE"),
        ):
            means = np.asarray([item["mean"] for item in records[name][method]])
            q16 = np.asarray([item["q16"] for item in records[name][method]])
            q84 = np.asarray([item["q84"] for item in records[name][method]])
            ax.loglog(
                shots_values,
                means,
                marker=marker,
                color=color,
                label=label,
            )
            ax.fill_between(shots_values, q16, q84, color=color, alpha=0.16)
        ax.set_title(titles[name])
        ax.set_xlabel("Shots per Pauli setting, N")
        ax.set_ylabel("Mean infidelity, 1 − F")
        ax.legend(frameon=False)

    ax = axes[3]
    ax.semilogx(
        shots_values,
        100 * np.asarray(records["plus_y"]["linear_nonphysical"]),
        "o-",
        color=COLORS["green"],
        label=r"$|+y\rangle$",
    )
    ax.semilogx(
        shots_values,
        100 * np.asarray(records["bell"]["linear_nonphysical"]),
        "s-",
        color=COLORS["purple"],
        label=r"$|\Phi^+\rangle$",
    )
    ax.semilogx(
        shots_values,
        100 * np.asarray(records["ghz4"]["linear_nonphysical"]),
        "^-",
        color=COLORS["red"],
        label=r"$|GHZ_4\rangle$",
    )
    ax.set_ylim(-3, 103)
    ax.set_title("Linear inversion leaves state space")
    ax.set_xlabel("Shots per Pauli setting, N")
    ax.set_ylabel("Nonphysical LI estimates (%)")
    ax.legend(frameon=False)
    fig.suptitle("Physical reconstruction improves with finite-shot data")
    paths = save_figure(fig, "poster_reconstruction_quality")
    results["reconstruction_quality"] = {
        "shots_per_setting": shots_values.tolist(),
        "trials": trials,
        "mle_tolerance": 2e-8,
        "records": records,
        "files": paths,
    }


def plot_mle_convergence(results):
    """Show monotone MLE optimization and the reconstructed Bell state."""
    target = reference_states()["bell"]
    dataset = generate_measurement_dataset(target, 256, seed=71)
    result = maximum_likelihood(dataset, max_iterations=500, tolerance=1e-11)
    history = np.asarray(result.objective_history)
    gap = np.maximum(history - history[-1], 1e-14)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.25))
    ax = axes[0]
    ax.semilogy(np.arange(len(gap)), gap, "o-", color=COLORS["blue"])
    ax.set_title("Backtracking decreases NLL")
    ax.set_xlabel("Accepted iteration")
    ax.set_ylabel(r"NLL $-$ final NLL")
    ax.text(
        0.05,
        0.08,
        f"converged={result.converged}\niterations={result.iterations}",
        transform=ax.transAxes,
    )

    for ax, matrix, title in (
        (axes[1], target, "True Bell density matrix"),
        (axes[2], np.asarray(result.rho), "MLE density matrix"),
    ):
        ax.imshow(np.abs(matrix), vmin=0, vmax=0.52, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.text(
            0.5,
            -0.18,
            r"color = $|\rho_{ij}|$, common scale 0–0.52",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color=COLORS["gray"],
        )
    fidelity = state_fidelity(target, result.rho)
    fig.suptitle(
        f"Multinomial MLE remains physical · fidelity={fidelity:.5f} · "
        f"min eigenvalue={result.min_eigenvalue:.1e}"
    )
    paths = save_figure(fig, "poster_mle_convergence")
    results["mle_convergence"] = {
        "shots_per_setting": 256,
        "seed": 71,
        "objective_history": history.tolist(),
        "converged": result.converged,
        "iterations": result.iterations,
        "fidelity": fidelity,
        "min_eigenvalue": result.min_eigenvalue,
        "files": paths,
    }


def _median_runtime(function, repeats=3):
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


def plot_backend_comparison(results):
    """Compare fixed-count NumPy and JAX MLE on the local CPU."""
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("JAX is required for this poster plot") from exc

    references = reference_states()
    states = (references["plus_y"], references["bell"])
    labels = ["1 qubit", "2 qubits"]
    numpy_times = []
    jax_times = []
    max_differences = []
    fidelities = []
    numpy_converged = []
    jax_converged = []
    numpy_iterations = []
    jax_iterations = []

    for index, target in enumerate(states):
        numpy_dataset = generate_measurement_dataset(target, 256, seed=91 + index)
        jax_dataset = generate_measurement_dataset(
            jnp.asarray(target), 256, seed=91 + index
        )

        # One warm-up run removes import and first primitive-dispatch overhead;
        # execution remains eager and includes all device synchronization.
        maximum_likelihood(
            numpy_dataset, max_iterations=300, tolerance=1e-8
        )
        maximum_likelihood(
            jax_dataset, max_iterations=300, tolerance=1e-8
        )
        numpy_time, numpy_result = _median_runtime(
            lambda: maximum_likelihood(
                numpy_dataset, max_iterations=300, tolerance=1e-8
            )
        )
        jax_time, jax_result = _median_runtime(
            lambda: maximum_likelihood(
                jax_dataset, max_iterations=300, tolerance=1e-8
            )
        )
        numpy_times.append(numpy_time)
        jax_times.append(jax_time)
        max_differences.append(
            float(np.max(np.abs(numpy_result.rho - np.asarray(jax_result.rho))))
        )
        fidelities.append(state_fidelity(target, numpy_result.rho))
        numpy_converged.append(numpy_result.converged)
        jax_converged.append(jax_result.converged)
        numpy_iterations.append(numpy_result.iterations)
        jax_iterations.append(jax_result.iterations)

    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axes[0]
    ax.bar(x - width / 2, np.asarray(numpy_times) * 1000, width,
           color=COLORS["blue"], label="NumPy CPU")
    ax.bar(x + width / 2, np.asarray(jax_times) * 1000, width,
           color=COLORS["red"], label="JAX CPU (eager)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median wall time (ms)")
    ax.set_title("Same reconstruction code, two backends")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.semilogy(x, max_differences, "o-", color=COLORS["green"])
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"max $|\rho_{NumPy}-\rho_{JAX}|$")
    ax.set_title("Fixed counts reproduce the same MLE")
    ax.set_ylim(min(max_differences) * 0.75, max(max_differences) * 1.7)
    for position, (difference, fidelity) in enumerate(
        zip(max_differences, fidelities)
    ):
        offset = (18, 12) if position == 0 else (-58, -45)
        ax.annotate(
            f"Δ={difference:.1e}\nF={fidelity:.5f}",
            (position, difference),
            xytext=offset,
            textcoords="offset points",
            ha="center",
        )
    fig.suptitle("Hardware-agnostic execution: accuracy agreement and local timing")
    paths = save_figure(fig, "poster_reconstruction_backends")
    results["backend_comparison"] = {
        "labels": labels,
        "numpy_seconds": numpy_times,
        "jax_seconds": jax_times,
        "max_density_matrix_difference": max_differences,
        "fidelity": fidelities,
        "mle_tolerance": 1e-8,
        "numpy_converged": numpy_converged,
        "jax_converged": jax_converged,
        "numpy_iterations": numpy_iterations,
        "jax_iterations": jax_iterations,
        "jax_platform": jax.default_backend(),
        "files": paths,
    }


def plot_resource_limit(results):
    """Make the dense n=20 limitation quantitatively explicit."""
    qubits = np.arange(1, 21)
    settings = 3.0**qubits
    density_tib = (16.0 * 4.0**qubits) / 2**40

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    ax = axes[0]
    ax.semilogy(qubits, settings, "o-", color=COLORS["green"])
    ax.set_title("Exhaustive Pauli settings")
    ax.set_xlabel("Qubits, n")
    ax.set_ylabel(r"Settings, $3^n$")
    ax.set_xticks([1, 5, 10, 15, 20])

    ax = axes[1]
    ax.semilogy(qubits, density_tib, "s-", color=COLORS["purple"])
    ax.axhline(1, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    ax.set_title("One dense complex128 density matrix")
    ax.set_xlabel("Qubits, n")
    ax.set_ylabel("Storage (TiB)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.annotate(
        f"n=20: {density_tib[-1]:.0f} TiB",
        (20, density_tib[-1]),
        xytext=(-75, -10),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"]},
    )
    fig.suptitle("Dense full-state tomography cannot scale to 20 qubits")
    paths = save_figure(fig, "poster_reconstruction_resource_limit")
    results["resource_limit"] = {
        "qubits": qubits.tolist(),
        "settings": settings.astype(np.int64).tolist(),
        "density_matrix_tib": density_tib.tolist(),
        "files": paths,
    }


def main():
    configure_style()
    results = {
        "metadata": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "ai_disclosure": AI_FOOTER,
        }
    }
    plot_reconstruction_quality(results)
    plot_mle_convergence(results)
    plot_backend_comparison(results)
    plot_resource_limit(results)
    data_path = OUTPUT_DIR / "poster_plot_data.json"
    with data_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"Generated 4 reconstruction figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
