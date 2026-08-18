"""Run the staged validation and scaling study used by the report/poster.

The script is deliberately deterministic and records its execution environment.
All benchmark timings are smoke-test timings, never portable performance claims.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from nbqst.denoise import project_density_matrix
from nbqst.measurements import exact_pauli_measurements, simulate_pauli_measurements
from nbqst.metrics import fidelity, hilbert_schmidt_distance, minimum_eigenvalue, purity
from nbqst.neural import load_neural_model, neural_state_reconstruction
from nbqst.noise import (
    amplitude_damping_channel,
    asymmetric_pauli_channel,
    coherent_rotation_channel,
    global_depolarizing_channel,
    phase_damping_channel,
)
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli
from nbqst.states import haar_random_pure


SEED = 20260818
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
JOB_ENVIRONMENT = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
)


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_command(arguments):
    try:
        result = subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def execution_manifest():
    blas = io.StringIO()
    with contextlib.redirect_stdout(blas):
        np.show_config()
    manifest = {
        "study_seed": SEED,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count_visible": os.cpu_count(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "thread_environment": {name: os.environ.get(name, "unset") for name in THREAD_ENVIRONMENT},
        "job_environment": {name: os.environ.get(name, "unset") for name in JOB_ENVIRONMENT},
        "backends": {
            name: bool(importlib.util.find_spec(name)) for name in ("numpy", "jax", "cupy")
        },
        "blas_configuration": blas.getvalue(),
    }
    mac_hardware = safe_command(("system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"))
    if mac_hardware:
        raw = json.loads(mac_hardware)
        hardware = (raw.get("SPHardwareDataType") or [{}])[0]
        display = (raw.get("SPDisplaysDataType") or [{}])[0]
        manifest["hardware"] = {
            "machine_name": hardware.get("machine_name"),
            "machine_model": hardware.get("machine_model"),
            "chip_type": hardware.get("chip_type"),
            "processor_description": hardware.get("number_processors"),
            "physical_memory": hardware.get("physical_memory"),
            "gpu_model": display.get("sppci_model"),
            "gpu_cores": display.get("sppci_cores"),
        }
    gpu_query = safe_command(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        )
    )
    if gpu_query:
        manifest["nvidia_gpus"] = gpu_query.splitlines()
    return manifest


def ks_distance_beta_one_d_minus_one(samples, dimension):
    ordered = np.sort(np.asarray(samples))
    cdf = 1.0 - (1.0 - ordered) ** (dimension - 1)
    count = len(ordered)
    upper = np.arange(1, count + 1) / count
    lower = np.arange(count) / count
    return float(max(np.max(np.abs(upper - cdf)), np.max(np.abs(cdf - lower))))


def verify_haar(rng, samples_per_size):
    rows = []
    for n_qubits in range(1, 6):
        dimension = 2**n_qubits
        overlaps = []
        maximum_trace_error = 0.0
        maximum_purity_error = 0.0
        for _ in range(samples_per_size):
            rho = haar_random_pure(n_qubits, rng=rng)
            overlaps.append(float(rho[0, 0].real))
            maximum_trace_error = max(maximum_trace_error, abs(float(np.trace(rho).real) - 1.0))
            maximum_purity_error = max(maximum_purity_error, abs(float(purity(rho)) - 1.0))
        empirical_mean = float(np.mean(overlaps))
        empirical_second = float(np.mean(np.square(overlaps)))
        expected_mean = 1.0 / dimension
        expected_second = 2.0 / (dimension * (dimension + 1.0))
        rows.append(
            {
                "n_qubits": n_qubits,
                "dimension": dimension,
                "samples": samples_per_size,
                "empirical_mean_overlap": empirical_mean,
                "expected_mean_overlap": expected_mean,
                "relative_mean_error": abs(empirical_mean - expected_mean) / expected_mean,
                "empirical_second_moment": empirical_second,
                "expected_second_moment": expected_second,
                "relative_second_moment_error": abs(empirical_second - expected_second) / expected_second,
                "ks_distance_to_beta_1_d_minus_1": ks_distance_beta_one_d_minus_one(overlaps, dimension),
                "maximum_trace_error": maximum_trace_error,
                "maximum_purity_error": maximum_purity_error,
            }
        )
    return rows


def verify_noise():
    plus = np.full((2, 2), 0.5, dtype=complex)
    excited = np.diag([0.0, 1.0]).astype(complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    checks = []

    depolarized = global_depolarizing_channel(plus, 0.2)
    checks.append(("depolarizing_bloch_shrink", float(np.trace(depolarized @ x).real), 0.8))
    damped = amplitude_damping_channel(excited, 0.3)
    checks.append(("amplitude_damping_excited_population", float(damped[1, 1].real), 0.7))
    dephased = phase_damping_channel(plus, 0.25)
    checks.append(("phase_damping_coherence", float(dephased[0, 1].real), 0.375))
    rotated = coherent_rotation_channel(plus, 0.7, axis="Z")
    checks.append(("coherent_rotation_purity", float(purity(rotated)), 1.0))
    biased = asymmetric_pauli_channel(plus, p_x=0.02, p_y=0.03, p_z=0.07)
    checks.append(("asymmetric_pauli_trace", float(np.trace(biased).real), 1.0))

    return [
        {
            "check": name,
            "observed": observed,
            "expected": expected,
            "absolute_error": abs(observed - expected),
            "passed_at_1e-12": abs(observed - expected) <= 1e-12,
        }
        for name, observed, expected in checks
    ]


def verify_exact_reconstruction(rng):
    rows = []
    for n_qubits in range(1, 6):
        truth = haar_random_pure(n_qubits, rng=rng)
        estimate = linear_inversion_pauli(exact_pauli_measurements(truth))
        rows.append(
            {
                "n_qubits": n_qubits,
                "frobenius_error": float(hilbert_schmidt_distance(truth, estimate)),
                "minimum_eigenvalue": float(minimum_eigenvalue(estimate)),
                "trace_error": abs(float(np.trace(estimate).real) - 1.0),
                "passed_at_1e-10": bool(np.allclose(truth, estimate, atol=1e-10)),
            }
        )
    return rows


def negative_eigenvalue_study(rng, replicates):
    shots_grid = (50, 100, 200, 500, 1000, 2000)
    rows = []
    for n_qubits in range(1, 6):
        for shots in shots_grid:
            counts = []
            minima = []
            for _ in range(replicates):
                truth = haar_random_pure(n_qubits, rng=rng)
                data = simulate_pauli_measurements(truth, shots, rng=rng)
                estimate = linear_inversion_pauli(data)
                eigenvalues = np.linalg.eigvalsh(estimate)
                counts.append(int(np.sum(eigenvalues < -1e-10)))
                minima.append(float(np.min(eigenvalues)))
            rows.append(
                {
                    "n_qubits": n_qubits,
                    "dimension": 2**n_qubits,
                    "shots_per_setting": shots,
                    "settings": 3**n_qubits,
                    "replicates": replicates,
                    "mean_negative_eigenvalues": float(np.mean(counts)),
                    "std_negative_eigenvalues": float(np.std(counts, ddof=1)),
                    "nonphysical_fraction": float(np.mean(np.asarray(counts) > 0)),
                    "mean_minimum_eigenvalue": float(np.mean(minima)),
                }
            )
    return rows


def noise_cases(truth):
    confusion = np.array([[0.97, 0.05], [0.03, 0.95]])
    return {
        "finite_shots_only": (truth, None),
        "global_depolarizing_p08": (global_depolarizing_channel(truth, 0.08), None),
        "amplitude_damping_p08": (amplitude_damping_channel(truth, 0.08), None),
        "phase_damping_p08": (phase_damping_channel(truth, 0.08), None),
        "biased_pauli_02_01_05": (
            asymmetric_pauli_channel(truth, p_x=0.02, p_y=0.01, p_z=0.05),
            None,
        ),
        "coherent_z_rotation_008rad": (coherent_rotation_channel(truth, 0.08, axis="Z"), None),
        "readout_assignment_3pct_5pct": (truth, confusion),
    }


def reconstruction_under_noise(rng, replicates, model_path):
    model = load_neural_model(model_path) if model_path.exists() else None
    rows = []
    for replicate in range(replicates):
        ideal = haar_random_pure(2, rng=rng)
        for case_name, (target, confusion) in noise_cases(ideal).items():
            data = simulate_pauli_measurements(
                target,
                500,
                rng=rng,
                readout_confusion=confusion,
            )
            start = time.perf_counter()
            linear = linear_inversion_pauli(data)
            linear_time = time.perf_counter() - start
            start = time.perf_counter()
            mle = factorized_mle(data, max_iter=60)
            mle_time = time.perf_counter() - start
            estimates = {
                "linear_inversion": (linear, linear_time),
                "maximum_likelihood": (mle, mle_time),
            }
            if model is not None:
                start = time.perf_counter()
                neural = neural_state_reconstruction(data, model)
                estimates["neural_network"] = (neural, time.perf_counter() - start)
            for method, (estimate, elapsed) in estimates.items():
                rows.append(
                    {
                        "replicate": replicate,
                        "noise_case": case_name,
                        "method": method,
                        "fidelity_to_reconstruction_target": float(fidelity(target, estimate)),
                        "fidelity_to_ideal_pre_noise_state": float(fidelity(ideal, estimate)),
                        "ideal_to_target_fidelity": float(fidelity(ideal, target)),
                        "hs_distance_to_target": float(hilbert_schmidt_distance(target, estimate)),
                        "minimum_eigenvalue": float(minimum_eigenvalue(estimate)),
                        "physical": float(minimum_eigenvalue(estimate)) >= -1e-10,
                        "inference_seconds": elapsed,
                    }
                )
    return rows


def summarize_reconstruction(rows):
    grouped = {}
    for row in rows:
        key = (row["noise_case"], row["method"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (case, method), values in sorted(grouped.items()):
        summary.append(
            {
                "noise_case": case,
                "method": method,
                "replicates": len(values),
                "mean_target_fidelity": float(np.mean([item["fidelity_to_reconstruction_target"] for item in values])),
                "std_target_fidelity": float(np.std([item["fidelity_to_reconstruction_target"] for item in values], ddof=1)),
                "mean_ideal_to_target_fidelity": float(np.mean([item["ideal_to_target_fidelity"] for item in values])),
                "physical_fraction": float(np.mean([item["physical"] for item in values])),
                "mean_inference_seconds": float(np.mean([item["inference_seconds"] for item in values])),
            }
        )
    return summary


def scaling_study(rng, maximum_qubits):
    rows = []
    for n_qubits in range(1, maximum_qubits + 1):
        truth = haar_random_pure(n_qubits, rng=rng)
        start = time.perf_counter()
        data = simulate_pauli_measurements(truth, 200, rng=rng)
        acquisition = time.perf_counter() - start
        start = time.perf_counter()
        estimate = linear_inversion_pauli(data)
        linear_time = time.perf_counter() - start
        mle_time = "not_run"
        mle_fidelity = "not_run"
        if n_qubits <= 4:
            start = time.perf_counter()
            mle = factorized_mle(data, max_iter=20)
            mle_time = time.perf_counter() - start
            mle_fidelity = float(fidelity(truth, mle))
        rows.append(
            {
                "n_qubits": n_qubits,
                "dimension": 2**n_qubits,
                "settings_3_to_n": 3**n_qubits,
                "outcomes_across_settings_6_to_n": 6**n_qubits,
                "density_real_parameters_4_to_n": 4**n_qubits,
                "copies_at_200_shots": 200 * 3**n_qubits,
                "dense_density_mebibytes_complex128": 16 * 4**n_qubits / 2**20,
                "dense_frequencies_mebibytes_float64": 8 * 6**n_qubits / 2**20,
                "measurement_simulation_seconds": acquisition,
                "linear_inversion_seconds": linear_time,
                "linear_fidelity": float(fidelity(truth, project_density_matrix(estimate))),
                "mle_20_iterations_seconds": mle_time,
                "mle_fidelity": mle_fidelity,
            }
        )
    return rows


def theoretical_scaling(maximum_qubits=14):
    return [
        {
            "n_qubits": n,
            "settings_3_to_n": 3**n,
            "frequencies_6_to_n": 6**n,
            "density_parameters_4_to_n": 4**n,
            "copies_at_500_shots": 500 * 3**n,
            "dense_density_gib_complex128": 16 * 4**n / 2**30,
            "dense_frequency_gib_float64": 8 * 6**n / 2**30,
        }
        for n in range(1, maximum_qubits + 1)
    ]


def create_figures(output, haar_rows, negative_rows, noise_summary, scaling_rows, theory_rows):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    navy, teal, orange, red, purple = "#0B2447", "#147D92", "#F29E4C", "#C84A4A", "#7356A8"

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), constrained_layout=True)
    n_values = [row["n_qubits"] for row in haar_rows]
    axes[0].plot(n_values, [100 * row["relative_mean_error"] for row in haar_rows], "o-", color=teal, label="mean")
    axes[0].plot(n_values, [100 * row["relative_second_moment_error"] for row in haar_rows], "s-", color=orange, label="second moment")
    axes[0].set(xlabel="qubits", ylabel="relative error (%)", title="Haar moment checks")
    axes[0].legend(frameon=False)
    axes[1].plot(n_values, [row["ks_distance_to_beta_1_d_minus_1"] for row in haar_rows], "o-", color=navy)
    axes[1].set(xlabel="qubits", ylabel="KS distance", title=r"Overlap law: Beta$(1,d-1)$")
    figure.savefig(output / "haar_verification.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), constrained_layout=True)
    colors = [navy, teal, orange, red, purple]
    for n_qubits, color in zip(range(1, 6), colors):
        selected = [row for row in negative_rows if row["n_qubits"] == n_qubits]
        shots = [row["shots_per_setting"] for row in selected]
        axes[0].plot(shots, [row["mean_negative_eigenvalues"] for row in selected], "o-", color=color, label=f"{n_qubits}q")
        axes[1].plot(shots, [100 * row["nonphysical_fraction"] for row in selected], "o-", color=color, label=f"{n_qubits}q")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.16)
        axis.set_xlabel("shots per Pauli setting")
    axes[0].set_ylabel("mean negative eigenvalue count")
    axes[0].set_title("LI physicality failure")
    axes[1].set_ylabel("nonphysical reconstructions (%)")
    axes[1].set_title("At least one negative eigenvalue")
    axes[1].legend(frameon=False, ncol=2)
    figure.savefig(output / "negative_eigenvalues_li.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    case_order = [
        "finite_shots_only",
        "global_depolarizing_p08",
        "amplitude_damping_p08",
        "phase_damping_p08",
        "biased_pauli_02_01_05",
        "coherent_z_rotation_008rad",
        "readout_assignment_3pct_5pct",
    ]
    labels = ["shots", "depol.", "T1", "phase", "biased\nPauli", "coherent", "readout"]
    methods = ["linear_inversion", "maximum_likelihood", "neural_network"]
    method_labels = ["LI", "MLE", "NN"]
    method_colors = [navy, teal, orange]
    lookup = {(row["noise_case"], row["method"]): row for row in noise_summary}
    figure, axis = plt.subplots(figsize=(9.2, 3.3), constrained_layout=True)
    x_values = np.arange(len(case_order))
    width = 0.24
    for index, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        values = [lookup[(case, method)]["mean_target_fidelity"] for case in case_order]
        errors = [lookup[(case, method)]["std_target_fidelity"] for case in case_order]
        axis.bar(x_values + (index - 1) * width, values, width, yerr=errors, color=color, label=label, capsize=2)
    axis.set_ylim(0.74, 1.01)
    axis.set_ylabel("fidelity to measured/noisy target")
    axis.set_xticks(x_values, labels)
    axis.set_title("Two-qubit reconstruction under unmodeled noise (500 shots/setting)")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.16)
    figure.savefig(output / "noise_reconstruction.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    n_empirical = [row["n_qubits"] for row in scaling_rows]
    axes[0].plot(n_empirical, [row["measurement_simulation_seconds"] for row in scaling_rows], "o-", color=teal, label="Born + shots")
    axes[0].plot(n_empirical, [row["linear_inversion_seconds"] for row in scaling_rows], "s-", color=navy, label="LI")
    mle_rows = [row for row in scaling_rows if row["mle_20_iterations_seconds"] != "not_run"]
    axes[0].plot([row["n_qubits"] for row in mle_rows], [row["mle_20_iterations_seconds"] for row in mle_rows], "^-", color=orange, label="MLE (20 iter.)")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="qubits", ylabel="wall time (s)", title="Completed CPU smoke test")
    axes[0].legend(frameon=False)
    theory_n = [row["n_qubits"] for row in theory_rows]
    axes[1].plot(theory_n, [row["settings_3_to_n"] for row in theory_rows], color=teal, label=r"settings $3^n$")
    axes[1].plot(theory_n, [row["frequencies_6_to_n"] for row in theory_rows], color=orange, label=r"frequencies $6^n$")
    axes[1].plot(theory_n, [row["density_parameters_4_to_n"] for row in theory_rows], color=navy, label=r"state params. $4^n$")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="qubits", ylabel="count", title="Generic dense tomography is exponential")
    axes[1].legend(frameon=False)
    figure.savefig(output / "scaling.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    manifest = execution_manifest()
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    haar_rows = verify_haar(rng, args.haar_samples)
    write_csv(output / "haar_verification.csv", haar_rows)
    noise_rows = verify_noise()
    write_csv(output / "noise_analytic_verification.csv", noise_rows)
    exact_rows = verify_exact_reconstruction(rng)
    write_csv(output / "exact_li_verification.csv", exact_rows)
    negative_rows = negative_eigenvalue_study(rng, args.negative_replicates)
    write_csv(output / "li_negative_eigenvalues.csv", negative_rows)
    model_path = Path(args.neural_model)
    reconstruction_rows = reconstruction_under_noise(rng, args.noise_replicates, model_path)
    write_csv(output / "noise_reconstruction.csv", reconstruction_rows)
    reconstruction_summary = summarize_reconstruction(reconstruction_rows)
    write_csv(output / "noise_reconstruction_summary.csv", reconstruction_summary)
    scaling_rows = scaling_study(rng, args.max_scaling_qubits)
    write_csv(output / "empirical_scaling.csv", scaling_rows)
    theory_rows = theoretical_scaling()
    write_csv(output / "theoretical_scaling.csv", theory_rows)
    create_figures(output, haar_rows, negative_rows, reconstruction_summary, scaling_rows, theory_rows)

    checks = {
        "haar_trace_purity": all(row["maximum_trace_error"] < 1e-12 and row["maximum_purity_error"] < 1e-12 for row in haar_rows),
        "haar_distribution_ks_under_0_05": all(row["ks_distance_to_beta_1_d_minus_1"] < 0.05 for row in haar_rows),
        "noise_analytic": all(row["passed_at_1e-12"] for row in noise_rows),
        "exact_li": all(row["passed_at_1e-10"] for row in exact_rows),
        "mle_physical_under_all_noise": all(row["physical_fraction"] == 1.0 for row in reconstruction_summary if row["method"] == "maximum_likelihood"),
        "neural_physical_under_all_noise": all(row["physical_fraction"] == 1.0 for row in reconstruction_summary if row["method"] == "neural_network"),
    }
    (output / "verification_status.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(f"one or more verification gates failed: {checks}")


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--output", default="results/verification_study")
    result.add_argument("--haar-samples", type=int, default=2500)
    result.add_argument("--negative-replicates", type=int, default=20)
    result.add_argument("--noise-replicates", type=int, default=8)
    result.add_argument("--max-scaling-qubits", type=int, default=6)
    result.add_argument("--neural-model", default="results/neural_comparison_model.npz")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
