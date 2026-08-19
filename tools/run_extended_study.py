"""Run the NBQSS hardware, shot-complexity, and classical-shadow studies.

The study has three deliberately separate claims:

1. synchronized wall-clock scaling for the backends actually available;
2. an empirical shot-grid search for 99% rank-one-MLE fidelity;
3. matched-copy observable estimation with local-Pauli classical shadows,
   LI, MLE, and an optional pretrained neural estimator.

Classical-shadow method: H.-Y. Huang, R. Kueng, and J. Preskill,
"Predicting many properties of a quantum system from very few measurements,"
Nature Physics 16, 1050-1057 (2020), arXiv:2002.08953.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from nbqst.backend import backend_runtime_metadata
from nbqst.experiment import benchmark, summarize as summarize_benchmark
from nbqst.measurements import simulate_pauli_measurements
from nbqst.metrics import fidelity, minimum_eigenvalue
from nbqst.neural import load_neural_model, neural_state_reconstruction
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli
from nbqst.shadows import (
    ClassicalShadowProtocol,
    PauliObservable,
    estimate_observable_from_measurements,
    observable_expectation,
)
from nbqst.states import haar_random_pure, random_product_state


SEED = 20260819
STATE_GENERATORS = {"product": random_product_state, "haar": haar_random_pure}


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_command(arguments):
    try:
        result = subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def backend_namespace(name):
    if name == "numpy":
        return np
    if name == "cupy":
        import cupy

        return cupy
    if name == "jax":
        import jax.numpy

        return jax.numpy
    raise ValueError(name)


def hardware_study(output_dir, *, backends, qubits, shots, states, mle_iterations, repeats):
    records = []
    availability = {}
    for name in backends:
        module = "numpy" if name == "numpy" else name
        installed = importlib.util.find_spec(module) is not None
        availability[name] = {"installed": installed, "executed": False}
        if not installed:
            availability[name]["reason"] = "Python package is unavailable in this environment"
            continue
        try:
            xp = backend_namespace(name)
            selected = benchmark(
                qubits=tuple(qubits),
                shots=(shots,),
                state_types=("haar",),
                states_per_case=states,
                xp=xp,
                seed=SEED,
                mle_iterations=mle_iterations,
                methods=("mle",),
                warmup_rounds=1,
                timing_repeats=repeats,
            )
        except Exception as error:  # keep unavailable accelerators explicit in the manifest
            availability[name]["reason"] = f"{type(error).__name__}: {error}"
            continue
        availability[name]["executed"] = True
        availability[name]["runtime"] = backend_runtime_metadata(xp)
        records.extend(selected)

    if records:
        write_csv(output_dir / "hardware_benchmark.csv", records)
        write_csv(output_dir / "hardware_benchmark_summary.csv", summarize_benchmark(records))

    theoretical = []
    for n_qubits in range(1, 21):
        theoretical.append(
            {
                "n_qubits": n_qubits,
                "hilbert_dimension": 2**n_qubits,
                "density_matrix_complex_values": 4**n_qubits,
                "density_matrix_complex128_bytes": 16 * (4**n_qubits),
                "complete_pauli_settings": 3**n_qubits,
                "outcomes_per_setting": 2**n_qubits,
                "stored_frequency_values": 6**n_qubits,
                "count_array_int64_bytes": 8 * (6**n_qubits),
            }
        )
    write_csv(output_dir / "dense_theoretical_scaling_1_to_20_qubits.csv", theoretical)
    return records, availability


def shot_requirement_study(
    output_dir,
    *,
    qubits,
    shots_grid,
    replicates,
    mle_iterations,
):
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for n_qubits in qubits:
        for state_type, generator in STATE_GENERATORS.items():
            for replicate in range(replicates):
                truth = generator(n_qubits, rng=rng)
                for shots in shots_grid:
                    data = simulate_pauli_measurements(truth, shots, rng=rng)
                    started = time.perf_counter_ns()
                    estimate, history = factorized_mle(
                        data,
                        rank=1,
                        max_iter=mle_iterations,
                        return_history=True,
                    )
                    elapsed = (time.perf_counter_ns() - started) / 1e9
                    value = float(fidelity(truth, estimate))
                    rows.append(
                        {
                            "n_qubits": n_qubits,
                            "state_type": state_type,
                            "replicate": replicate,
                            "shots_per_setting": shots,
                            "settings": 3**n_qubits,
                            "total_copies": shots * (3**n_qubits),
                            "method": "rank_one_mle",
                            "mle_iterations_requested": mle_iterations,
                            "mle_iterations_completed": max(0, len(history) - 1),
                            "fidelity": value,
                            "hit_99_percent": value >= 0.99,
                            "reconstruction_seconds": elapsed,
                        }
                    )
    write_csv(output_dir / "shot_requirement_raw.csv", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["n_qubits"], row["state_type"], row["shots_per_setting"])].append(row)
    summary = []
    for (n_qubits, state_type, shots), selected in sorted(grouped.items()):
        values = np.asarray([row["fidelity"] for row in selected])
        summary.append(
            {
                "n_qubits": n_qubits,
                "state_type": state_type,
                "shots_per_setting": shots,
                "total_copies": shots * (3**n_qubits),
                "replicates": len(selected),
                "mean_fidelity": float(np.mean(values)),
                "median_fidelity": float(np.median(values)),
                "p10_fidelity": float(np.percentile(values, 10)),
                "p90_fidelity": float(np.percentile(values, 90)),
                "success_fraction_99": float(np.mean(values >= 0.99)),
                "mean_reconstruction_seconds": float(
                    np.mean([row["reconstruction_seconds"] for row in selected])
                ),
            }
        )
    write_csv(output_dir / "shot_requirement_summary.csv", summary)

    requirements = []
    for n_qubits in qubits:
        for state_type in STATE_GENERATORS:
            selected = [
                row for row in summary if row["n_qubits"] == n_qubits and row["state_type"] == state_type
            ]
            median_hits = [row for row in selected if row["median_fidelity"] >= 0.99]
            majority_hits = [row for row in selected if row["success_fraction_99"] >= 0.5]
            requirements.append(
                {
                    "n_qubits": n_qubits,
                    "state_type": state_type,
                    "criterion": "median_fidelity_at_least_0.99",
                    "required_shots_per_setting": (
                        min(row["shots_per_setting"] for row in median_hits) if median_hits else "not_reached"
                    ),
                    "required_total_copies": (
                        min(row["total_copies"] for row in median_hits) if median_hits else "not_reached"
                    ),
                }
            )
            requirements.append(
                {
                    "n_qubits": n_qubits,
                    "state_type": state_type,
                    "criterion": "at_least_half_of_states_hit_0.99",
                    "required_shots_per_setting": (
                        min(row["shots_per_setting"] for row in majority_hits) if majority_hits else "not_reached"
                    ),
                    "required_total_copies": (
                        min(row["total_copies"] for row in majority_hits) if majority_hits else "not_reached"
                    ),
                }
            )
    write_csv(output_dir / "shot_requirement_thresholds.csv", requirements)
    return rows, summary, requirements


def observable_study(
    output_dir,
    *,
    budgets,
    replicates,
    mle_iterations,
    neural_model_path,
):
    n_qubits = 2
    observables = tuple(PauliObservable(label) for label in ("ZI", "IZ", "ZZ", "XX"))
    shadow_protocol = ClassicalShadowProtocol(median_of_means_groups=1)
    neural_model = load_neural_model(neural_model_path) if neural_model_path.exists() else None
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for state_type, generator in STATE_GENERATORS.items():
        for replicate in range(replicates):
            truth = generator(n_qubits, rng=rng)
            exact = {observable.label: observable_expectation(truth, observable) for observable in observables}
            for budget in budgets:
                shots_per_setting = max(1, budget // (3**n_qubits))
                qst_copies = shots_per_setting * (3**n_qubits)
                shadow = shadow_protocol.acquire(truth, budget, rng=rng)
                shadow_estimates = {
                    estimate.observable.label: estimate for estimate in shadow_protocol.estimate_many(shadow, observables)
                }
                data = simulate_pauli_measurements(truth, shots_per_setting, rng=rng)
                started = time.perf_counter_ns()
                linear = linear_inversion_pauli(data)
                linear_seconds = (time.perf_counter_ns() - started) / 1e9
                started = time.perf_counter_ns()
                mle = factorized_mle(data, rank=1, max_iter=mle_iterations)
                mle_seconds = (time.perf_counter_ns() - started) / 1e9
                estimates = {
                    "linear_inversion": (linear, linear_seconds),
                    "rank_one_mle": (mle, mle_seconds),
                }
                if neural_model is not None:
                    started = time.perf_counter_ns()
                    neural = neural_state_reconstruction(data, neural_model)
                    neural_seconds = (time.perf_counter_ns() - started) / 1e9
                    estimates["neural_network"] = (neural, neural_seconds)

                for observable in observables:
                    target = exact[observable.label]
                    shadow_value = shadow_estimates[observable.label].value
                    rows.append(
                        {
                            "n_qubits": n_qubits,
                            "state_type": state_type,
                            "replicate": replicate,
                            "observable": observable.label,
                            "observable_weight": observable.weight,
                            "method": "classical_shadow",
                            "requested_copy_budget": budget,
                            "actual_total_copies": budget,
                            "collection_total_copies": budget,
                            "shots_per_qst_setting": "not_applicable",
                            "exact_expectation": target,
                            "estimated_expectation": shadow_value,
                            "error": shadow_value - target,
                            "absolute_error": abs(shadow_value - target),
                            "squared_error": (shadow_value - target) ** 2,
                            "estimate_standard_error": shadow_estimates[observable.label].standard_error,
                            "estimate_physical": "not_applicable",
                            "inference_seconds": "not_recorded_separately",
                        }
                    )
                    targeted_setting = "".join(
                        character if character != "I" else "Z" for character in observable.label
                    )
                    targeted_data = simulate_pauli_measurements(
                        truth,
                        budget,
                        settings=(targeted_setting,),
                        rng=rng,
                    )
                    targeted = estimate_observable_from_measurements(targeted_data, observable)
                    rows.append(
                        {
                            "n_qubits": n_qubits,
                            "state_type": state_type,
                            "replicate": replicate,
                            "observable": observable.label,
                            "observable_weight": observable.weight,
                            "method": "direct_targeted_pauli",
                            "requested_copy_budget": budget,
                            "actual_total_copies": budget,
                            "collection_total_copies": budget * len(observables),
                            "shots_per_qst_setting": "one_targeted_setting",
                            "exact_expectation": target,
                            "estimated_expectation": targeted.value,
                            "error": targeted.value - target,
                            "absolute_error": abs(targeted.value - target),
                            "squared_error": (targeted.value - target) ** 2,
                            "estimate_standard_error": targeted.standard_error,
                            "estimate_physical": "not_applicable",
                            "inference_seconds": "not_recorded_separately",
                        }
                    )
                    split_shots = max(1, budget // len(observables))
                    split_data = simulate_pauli_measurements(
                        truth,
                        split_shots,
                        settings=(targeted_setting,),
                        rng=rng,
                    )
                    split_targeted = estimate_observable_from_measurements(split_data, observable)
                    rows.append(
                        {
                            "n_qubits": n_qubits,
                            "state_type": state_type,
                            "replicate": replicate,
                            "observable": observable.label,
                            "observable_weight": observable.weight,
                            "method": "direct_split_collection_budget",
                            "requested_copy_budget": budget,
                            "actual_total_copies": split_shots,
                            "collection_total_copies": split_shots * len(observables),
                            "shots_per_qst_setting": "one_targeted_setting_per_observable",
                            "exact_expectation": target,
                            "estimated_expectation": split_targeted.value,
                            "error": split_targeted.value - target,
                            "absolute_error": abs(split_targeted.value - target),
                            "squared_error": (split_targeted.value - target) ** 2,
                            "estimate_standard_error": split_targeted.standard_error,
                            "estimate_physical": "not_applicable",
                            "inference_seconds": "not_recorded_separately",
                        }
                    )
                    for method, (estimate, elapsed) in estimates.items():
                        value = observable_expectation(estimate, observable)
                        minimum = float(minimum_eigenvalue(estimate))
                        rows.append(
                            {
                                "n_qubits": n_qubits,
                                "state_type": state_type,
                                "replicate": replicate,
                                "observable": observable.label,
                                "observable_weight": observable.weight,
                                "method": method,
                                "requested_copy_budget": budget,
                                "actual_total_copies": qst_copies,
                                "collection_total_copies": qst_copies,
                                "shots_per_qst_setting": shots_per_setting,
                                "exact_expectation": target,
                                "estimated_expectation": value,
                                "error": value - target,
                                "absolute_error": abs(value - target),
                                "squared_error": (value - target) ** 2,
                                "estimate_standard_error": "not_available",
                                "estimate_physical": minimum >= -1e-10,
                                "inference_seconds": elapsed,
                            }
                        )
    write_csv(output_dir / "observable_estimation_raw.csv", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["state_type"], row["observable"], row["observable_weight"], row["method"], row["requested_copy_budget"])].append(row)
    summary = []
    for (state_type, observable, weight, method, budget), selected in sorted(grouped.items()):
        errors = np.asarray([float(row["error"]) for row in selected])
        summary.append(
            {
                "n_qubits": n_qubits,
                "state_type": state_type,
                "observable": observable,
                "observable_weight": weight,
                "method": method,
                "requested_copy_budget": budget,
                "mean_actual_total_copies": float(
                    np.mean([float(row["actual_total_copies"]) for row in selected])
                ),
                "mean_collection_total_copies": float(
                    np.mean([float(row["collection_total_copies"]) for row in selected])
                ),
                "replicates": len(selected),
                "bias": float(np.mean(errors)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "p90_absolute_error": float(np.percentile(np.abs(errors), 90)),
                "physical_fraction": (
                    float(np.mean([row["estimate_physical"] for row in selected]))
                    if method not in {
                        "classical_shadow",
                        "direct_targeted_pauli",
                        "direct_split_collection_budget",
                    }
                    else "not_applicable"
                ),
            }
        )
    write_csv(output_dir / "observable_estimation_summary.csv", summary)
    return rows, summary


def make_plots(output_dir, *, hardware_records, shot_summary, observable_summary):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    colors = {
        "numpy": "#1f5b99",
        "cupy": "#e76f51",
        "jax": "#2a9d8f",
        "product": "#2a9d8f",
        "haar": "#7b2cbf",
        "classical_shadow": "#e76f51",
        "linear_inversion": "#577590",
        "rank_one_mle": "#2a9d8f",
        "neural_network": "#7b2cbf",
        "direct_targeted_pauli": "#222222",
        "direct_split_collection_budget": "#9c6644",
    }

    if hardware_records:
        summary = summarize_benchmark(hardware_records)
        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for backend in sorted({row["backend"] for row in summary}):
            selected = sorted((row for row in summary if row["backend"] == backend), key=lambda row: row["n_qubits"])
            axis.plot(
                [row["n_qubits"] for row in selected],
                [row["median_reconstruction_seconds"] for row in selected],
                marker="o",
                linewidth=2,
                color=colors.get(backend),
                label=backend,
            )
        axis.set_yscale("log")
        axis.set_xlabel("qubits")
        axis.set_ylabel("median synchronized MLE reconstruction time (s)")
        axis.set_title("Dense MLE scaling on backends available to this run")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.savefig(output_dir / "hardware_mle_scaling.png", dpi=200)
        plt.close(figure)

    figure, axes = plt.subplots(1, len({row["n_qubits"] for row in shot_summary}), figsize=(12, 4), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, n_qubits in zip(axes, sorted({row["n_qubits"] for row in shot_summary})):
        for state_type in STATE_GENERATORS:
            selected = sorted(
                (row for row in shot_summary if row["n_qubits"] == n_qubits and row["state_type"] == state_type),
                key=lambda row: row["total_copies"],
            )
            axis.plot(
                [row["total_copies"] for row in selected],
                [row["median_fidelity"] for row in selected],
                marker="o",
                color=colors[state_type],
                label=state_type,
            )
            axis.fill_between(
                [row["total_copies"] for row in selected],
                [row["p10_fidelity"] for row in selected],
                [row["p90_fidelity"] for row in selected],
                color=colors[state_type],
                alpha=0.12,
            )
        axis.axhline(0.99, color="#222222", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_ylim(0.88, 1.002)
        axis.set_title(f"{n_qubits} qubit{'s' if n_qubits > 1 else ''}")
        axis.set_xlabel("total measured copies")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("rank-one MLE fidelity")
    axes[-1].legend(frameon=False)
    figure.suptitle("Shot grid for the 99% fidelity target")
    figure.savefig(output_dir / "shot_requirement_99_fidelity.png", dpi=200)
    plt.close(figure)

    states = tuple(STATE_GENERATORS)
    observables = sorted({row["observable"] for row in observable_summary})
    figure, axes = plt.subplots(len(states), len(observables), figsize=(14, 6.5), constrained_layout=True, sharex=True)
    for row_index, state_type in enumerate(states):
        for column, observable in enumerate(observables):
            axis = axes[row_index, column]
            selected = [
                row for row in observable_summary if row["state_type"] == state_type and row["observable"] == observable
            ]
            plot_methods = {
                "classical_shadow",
                "direct_split_collection_budget",
                "linear_inversion",
                "rank_one_mle",
                "neural_network",
            }
            for method in sorted({row["method"] for row in selected} & plot_methods):
                series = sorted((row for row in selected if row["method"] == method), key=lambda row: row["requested_copy_budget"])
                axis.plot(
                    [row["requested_copy_budget"] for row in series],
                    [row["rmse"] for row in series],
                    marker="o",
                    linewidth=1.7,
                    color=colors.get(method),
                    label=method.replace("_", " "),
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(f"{state_type} · {observable}")
            axis.grid(alpha=0.2)
            if row_index == len(states) - 1:
                axis.set_xlabel("measured copies")
            if column == 0:
                axis.set_ylabel("observable RMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle("Matched-budget observable estimation: shadows versus full QST")
    figure.savefig(output_dir / "classical_shadows_observable_rmse.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def manifest(args, availability):
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    scheduler_names = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
    )
    result = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "command": " ".join(sys.argv),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus_visible": os.cpu_count(),
            "python": sys.version.replace("\n", " "),
            "numpy": np.__version__,
            "thread_environment": {name: os.environ.get(name, "unset") for name in thread_names},
            "scheduler_environment": {name: os.environ.get(name, "unset") for name in scheduler_names},
        },
        "backend_availability": availability,
        "configuration": vars(args),
        "scientific_definitions": {
            "shot_requirement": "smallest tested grid point whose median fidelity is at least 0.99; a second table records >=50% state success",
            "qst_copy_cost": "shots_per_setting times 3**n complete local-Pauli settings",
            "shadow_copy_cost": "one independently randomized local-Pauli measurement per snapshot",
            "observable_comparison": "root mean squared error at matched requested state-copy budgets",
            "mle_model": "rank-one factorized multinomial maximum likelihood for pure product and Haar states",
        },
        "limitations": [
            "Only installed and executable backends produce timing records.",
            "A package-presence check is not evidence of GPU execution; inspect device_platform and device_name.",
            "Thresholds are discrete-grid empirical estimates, not asymptotic sample-complexity theorems.",
            "The bundled neural model is fixed, two-qubit, and trained at 500 shots per setting.",
        ],
    }
    hardware_json = safe_command(("system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"))
    if hardware_json:
        raw = json.loads(hardware_json)
        hardware = (raw.get("SPHardwareDataType") or [{}])[0]
        display = (raw.get("SPDisplaysDataType") or [{}])[0]
        result["host"]["hardware"] = {
            "machine_name": hardware.get("machine_name"),
            "machine_model": hardware.get("machine_model"),
            "chip_type": hardware.get("chip_type"),
            "processor_description": hardware.get("number_processors"),
            "physical_memory": hardware.get("physical_memory"),
            "gpu_model": display.get("sppci_model"),
            "gpu_cores": display.get("sppci_cores"),
        }
    nvidia = safe_command(
        ("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader")
    )
    if nvidia:
        result["host"]["nvidia_gpus"] = nvidia.splitlines()
    return result


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/extended_study"))
    parser.add_argument("--backends", nargs="+", choices=("numpy", "cupy", "jax"), default=("numpy", "cupy", "jax"))
    parser.add_argument("--hardware-qubits", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--hardware-shots", type=int, default=200)
    parser.add_argument("--hardware-states", type=int, default=2)
    parser.add_argument("--hardware-mle-iterations", type=int, default=8)
    parser.add_argument("--hardware-repeats", type=int, default=3)
    parser.add_argument("--shot-qubits", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--shot-grid", nargs="+", type=int, default=(5, 10, 20, 50, 100, 300, 1000))
    parser.add_argument("--shot-replicates", type=int, default=8)
    parser.add_argument("--shot-mle-iterations", type=int, default=30)
    parser.add_argument("--observable-budgets", nargs="+", type=int, default=(90, 180, 450, 900, 1800, 4500))
    parser.add_argument("--observable-replicates", type=int, default=20)
    parser.add_argument("--observable-mle-iterations", type=int, default=30)
    parser.add_argument("--neural-model", type=Path, default=Path("results/neural_comparison_model.npz"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hardware_records, availability = hardware_study(
        args.output_dir,
        backends=args.backends,
        qubits=args.hardware_qubits,
        shots=args.hardware_shots,
        states=args.hardware_states,
        mle_iterations=args.hardware_mle_iterations,
        repeats=args.hardware_repeats,
    )
    _, shot_summary, _ = shot_requirement_study(
        args.output_dir,
        qubits=args.shot_qubits,
        shots_grid=args.shot_grid,
        replicates=args.shot_replicates,
        mle_iterations=args.shot_mle_iterations,
    )
    _, observable_summary = observable_study(
        args.output_dir,
        budgets=args.observable_budgets,
        replicates=args.observable_replicates,
        mle_iterations=args.observable_mle_iterations,
        neural_model_path=args.neural_model,
    )
    make_plots(
        args.output_dir,
        hardware_records=hardware_records,
        shot_summary=shot_summary,
        observable_summary=observable_summary,
    )
    (args.output_dir / "extended_study_manifest.json").write_text(
        json.dumps(manifest(args, availability), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Extended study written to {args.output_dir}")


if __name__ == "__main__":
    main()
