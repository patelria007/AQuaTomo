"""Generate reproducible full-stack validation data and report figures.

This is analysis code, not hardware-agnostic core logic; NumPy and Matplotlib
are intentionally used for aggregation and rendering.

AI disclosure: this script and its figure design were generated with OpenAI
Codex assistance on 2026-08-17. Independently review the experiment and plots
before treating them as verified scientific results.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import jax
import matplotlib
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbqs_qst.measurement_generation import generate_measurement_dataset  # noqa: E402
from nbqs_qst.state_generation import (  # noqa: E402
    random_haar_state,
    random_product_state,
    random_state_with_purity,
)
from nbqs_qst.state_reconstruction import reconstruct, state_fidelity  # noqa: E402


OUTPUT = ROOT / "docs" / "validation_artifacts"
SHOT_COUNTS = (64, 256, 1024)
TRIALS = 12
AI_FOOTER = "AI-assisted figure · fixed seeds · independent review pending"
FAMILIES = (
    ("Product", lambda like: random_product_state(like, 2, seed=701)),
    ("Haar pure", lambda like: random_haar_state(like, 2, seed=702)),
    (
        "Mixed (purity 0.55)",
        lambda like: random_state_with_purity(like, 2, 0.55, seed=703),
    ),
)


def _quantiles(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "q16": float(np.quantile(array, 0.16)),
        "q84": float(np.quantile(array, 0.84)),
    }


def collect_quality_data():
    records = []
    for family_index, (family, generate) in enumerate(FAMILIES):
        target = generate(np.asarray(0.0))
        for shots in SHOT_COUNTS:
            per_method = {
                "linear": {"minimum_eigenvalue": [], "runtime_seconds": []},
                "pls": {
                    "fidelity": [],
                    "minimum_eigenvalue": [],
                    "runtime_seconds": [],
                },
                "mle": {
                    "fidelity": [],
                    "minimum_eigenvalue": [],
                    "runtime_seconds": [],
                    "converged": [],
                    "iterations": [],
                },
            }
            for trial in range(TRIALS):
                seed = 8000 + 1000 * family_index + trial
                dataset = generate_measurement_dataset(
                    target.rho, shots=shots, seed=seed
                )
                for method in ("linear", "pls", "mle"):
                    kwargs = {"initial": "pls"} if method == "mle" else {}
                    started = time.perf_counter()
                    result = reconstruct(dataset, method=method, **kwargs)
                    runtime = time.perf_counter() - started
                    values = per_method[method]
                    values["minimum_eigenvalue"].append(result.min_eigenvalue)
                    values["runtime_seconds"].append(runtime)
                    if method != "linear":
                        values["fidelity"].append(
                            state_fidelity(target.rho, result.rho)
                        )
                    if method == "mle":
                        values["converged"].append(result.converged)
                        values["iterations"].append(result.iterations)

            methods = {}
            for method, values in per_method.items():
                method_record = {
                    "minimum_eigenvalue": _quantiles(
                        values["minimum_eigenvalue"]
                    ),
                    "runtime_seconds": _quantiles(values["runtime_seconds"]),
                    "nonphysical_fraction": float(
                        np.mean(np.asarray(values["minimum_eigenvalue"]) < -1e-10)
                    ),
                }
                if method != "linear":
                    infidelities = 1.0 - np.asarray(values["fidelity"])
                    method_record["fidelity"] = _quantiles(values["fidelity"])
                    method_record["infidelity"] = _quantiles(infidelities)
                if method == "mle":
                    method_record["convergence_fraction"] = float(
                        np.mean(values["converged"])
                    )
                    method_record["iterations"] = _quantiles(values["iterations"])
                methods[method] = method_record

            records.append(
                {
                    "family": family,
                    "shots_per_setting": shots,
                    "trials": TRIALS,
                    "methods": methods,
                }
            )
    return records


def collect_backend_data():
    records = []
    for family, generate in FAMILIES:
        numpy_target = generate(np.asarray(0.0))
        jax_target = generate(jnp.asarray(0.0))
        numpy_dataset = generate_measurement_dataset(
            numpy_target.rho, shots=256, seed=12001
        )
        jax_dataset = generate_measurement_dataset(
            jax_target.rho, shots=256, seed=12001
        )
        count_difference = max(
            float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
            for left, right in zip(numpy_dataset.counts, jax_dataset.counts)
        )
        differences = {
            "target": float(
                np.max(np.abs(numpy_target.rho - np.asarray(jax_target.rho)))
            ),
            "counts": count_difference,
        }
        for method in ("linear", "pls", "mle"):
            kwargs = {"initial": "pls"} if method == "mle" else {}
            numpy_result = reconstruct(numpy_dataset, method=method, **kwargs)
            jax_result = reconstruct(jax_dataset, method=method, **kwargs)
            differences[method] = float(
                np.max(
                    np.abs(numpy_result.rho - np.asarray(jax_result.rho))
                )
            )
        records.append({"family": family, "max_absolute_difference": differences})
    return records


def plot_fidelity(quality):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharey=True)
    styles = {"pls": ("o", "#1f77b4"), "mle": ("s", "#d95f02")}
    for axis, (family, _) in zip(axes, FAMILIES):
        rows = [row for row in quality if row["family"] == family]
        for method, label in (("pls", "PLS"), ("mle", "MLE")):
            x = np.asarray([row["shots_per_setting"] for row in rows])
            mean = np.asarray(
                [row["methods"][method]["infidelity"]["mean"] for row in rows]
            )
            low = np.asarray(
                [row["methods"][method]["infidelity"]["q16"] for row in rows]
            )
            high = np.asarray(
                [row["methods"][method]["infidelity"]["q84"] for row in rows]
            )
            marker, color = styles[method]
            axis.plot(x, mean, marker=marker, color=color, label=label)
            axis.fill_between(x, np.maximum(low, 1e-7), high, color=color, alpha=0.18)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Shots per Pauli setting")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Mean infidelity, 1 − F")
    axes[-1].legend(frameon=False)
    fig.suptitle("Full-stack reconstruction quality across representative states")
    fig.text(0.5, 0.005, AI_FOOTER, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT / "full_stack_fidelity.png", dpi=190)
    fig.savefig(OUTPUT / "full_stack_fidelity.pdf")
    plt.close(fig)


def plot_physicality(quality):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), sharey=True)
    styles = {
        "linear": ("o", "#7f3c8d", "Linear inversion"),
        "pls": ("^", "#11a579", "PLS"),
        "mle": ("s", "#d95f02", "MLE"),
    }
    for axis, (family, _) in zip(axes, FAMILIES):
        rows = [row for row in quality if row["family"] == family]
        x = [row["shots_per_setting"] for row in rows]
        for method, (marker, color, label) in styles.items():
            y = [100 * row["methods"][method]["nonphysical_fraction"] for row in rows]
            axis.plot(x, y, marker=marker, color=color, label=label)
        axis.set_xscale("log", base=2)
        axis.set_ylim(-3, 103)
        axis.set_title(family)
        axis.set_xlabel("Shots per Pauli setting")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Nonphysical estimates (%)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Positive-semidefinite constraint remains active after sampling")
    fig.text(0.5, 0.005, AI_FOOTER, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT / "full_stack_physicality.png", dpi=190)
    fig.savefig(OUTPUT / "full_stack_physicality.pdf")
    plt.close(fig)


def plot_backend_agreement(backend):
    stages = ("target", "counts", "linear", "pls", "mle")
    values = np.asarray(
        [
            [row["max_absolute_difference"][stage] for stage in stages]
            for row in backend
        ]
    )
    display = np.log10(np.maximum(values, 1e-18))
    fig, axis = plt.subplots(figsize=(8.4, 3.8))
    image = axis.imshow(display, aspect="auto", vmin=-18, vmax=-9, cmap="viridis")
    axis.set_xticks(range(len(stages)), ["Target", "Counts", "Linear", "PLS", "MLE"])
    axis.set_yticks(range(len(FAMILIES)), [family for family, _ in FAMILIES])
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = "exact" if value == 0 else f"{value:.1e}"
            axis.text(column_index, row_index, label, ha="center", va="center", color="white")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("log10(max absolute difference), floor 10⁻¹⁸")
    axis.set_title("NumPy–JAX agreement through the full pipeline (fixed seeds)")
    fig.text(0.5, 0.005, AI_FOOTER, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT / "backend_agreement.png", dpi=190)
    fig.savefig(OUTPUT / "backend_agreement.pdf")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    quality = collect_quality_data()
    backend = collect_backend_data()
    payload = {
        "experiment": {
            "num_qubits": 2,
            "shot_counts": SHOT_COUNTS,
            "trials_per_point": TRIALS,
            "quality_backend": "NumPy",
            "backend_comparison": ["NumPy", "JAX (CPU in this environment)"],
            "randomness": "stdlib random.Random through project APIs",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "jax": jax.__version__,
            "jax_x64": bool(jax.config.jax_enable_x64),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "quality": quality,
        "backend_agreement": backend,
        "ai_disclosure": (
            "AI-assisted experiment code, figure design, and text; independent "
            "review pending."
        ),
    }
    (OUTPUT / "full_stack_validation_data.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    plot_fidelity(quality)
    plot_physicality(quality)
    plot_backend_agreement(backend)
    print(OUTPUT)


if __name__ == "__main__":
    main()
