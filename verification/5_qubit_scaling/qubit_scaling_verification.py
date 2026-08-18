"""Measure NumPy runtime and memory scaling with qubit count.

Run from the repository root with::

    python verification/5_qubit_scaling/qubit_scaling_verification.py

Each qubit/stage case runs in a fresh subprocess so that allocator state from
one case cannot contaminate the next case's peak-RSS measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import psutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nbqst.denoise import project_density_matrix  # noqa: E402
from nbqst.measurements import (  # noqa: E402
    MeasurementData,
    complete_pauli_settings,
    pauli_probabilities,
    simulate_pauli_measurements,
)
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli  # noqa: E402
from nbqst.states import haar_random_pure  # noqa: E402


DEFAULT_MAX_QUBITS = 10
DEFAULT_KERNEL_MAX_QUBITS = 6
DEFAULT_MLE_MAX_QUBITS = 6
DEFAULT_REPEATS = 3
DEFAULT_WARMUPS = 1
DEFAULT_SHOTS = 1024
DEFAULT_MLE_ITERATIONS = 3
DEFAULT_SEED = 20261001
DEFAULT_OUTPUT = Path(__file__).with_name("qubit_scaling_verification.png")
DEFAULT_JSON = Path(__file__).with_name("qubit_scaling_results.json")
STAGES = (
    "state_generation",
    "single_setting",
    "measurement_storage",
    "full_measurement",
    "linear_inversion",
    "fixed_iteration_mle",
)
STAGE_LABELS = {
    "state_generation": "state generation",
    "single_setting": "one Born setting",
    "measurement_storage": "measurement storage",
    "full_measurement": "all Pauli settings",
    "linear_inversion": "linear inversion",
    "fixed_iteration_mle": "fixed-iteration MLE",
}
STAGE_COLORS = {
    "state_generation": "#3569a8",
    "single_setting": "#6b8fd3",
    "measurement_storage": "#2a9d8f",
    "full_measurement": "#cf6a32",
    "linear_inversion": "#9b59b6",
    "fixed_iteration_mle": "#3a8b5b",
}
RESULT_PREFIX = "QUBIT_SCALING_RESULT="
MIB = 1024.0**2


def native_nbytes(value: Any) -> int:
    """Count persistent NumPy payload bytes without double-counting history."""
    if isinstance(value, MeasurementData):
        return sum(int(counts.nbytes) for counts in value.counts.values())
    if isinstance(value, tuple):
        return native_nbytes(value[0])
    return int(getattr(value, "nbytes", 0))


class PeakRssTracker:
    """Poll this worker's resident set while one complete case is active."""

    def __init__(self, interval_seconds: float = 0.002):
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.baseline = int(self.process.memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, int(self.process.memory_info().rss))
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.peak = max(self.peak, int(self.process.memory_info().rss))
        self._stop.set()
        self._thread.join()


def setting_for(n_qubits: int) -> str:
    axes = "XYZ"
    return "".join(axes[index % len(axes)] for index in range(n_qubits))


def allocate_measurement_storage(n_qubits: int) -> MeasurementData:
    """Allocate the complete int64 count representation without simulating it."""
    outcomes = 2**n_qubits
    counts = {
        setting: np.zeros(outcomes, dtype=np.int64)
        for setting in complete_pauli_settings(n_qubits)
    }
    return MeasurementData(n_qubits, counts, shots_per_setting=0)


def timed_samples(
    operation: Callable[[], Any],
    warmups: int,
    repeats: int,
) -> tuple[list[float], Any]:
    result = None
    for _ in range(warmups):
        result = operation()
    result = None
    gc.collect()

    samples = []
    for _ in range(repeats):
        result = None
        gc.collect()
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
    return samples, result


def validate_result(stage: str, result: Any, n_qubits: int, shots: int) -> dict[str, Any]:
    dim = 2**n_qubits
    if stage == "state_generation":
        if result.shape != (dim, dim) or not np.all(np.isfinite(result)):
            raise AssertionError("state generation returned an invalid array")
        return {"trace_error": float(abs(np.trace(result) - 1.0))}

    if stage == "single_setting":
        if result.shape != (dim,) or np.min(result) < -1e-12:
            raise AssertionError("Born probabilities are invalid")
        return {"probability_sum_error": float(abs(result.sum() - 1.0))}

    if stage == "measurement_storage":
        if len(result.counts) != 3**n_qubits:
            raise AssertionError("measurement storage has the wrong setting count")
        if any(np.any(counts) for counts in result.counts.values()):
            raise AssertionError("newly allocated count storage is not zero initialized")
        return {"settings": len(result.counts), "all_counts_zero": True}

    if stage == "full_measurement":
        if len(result.counts) != 3**n_qubits:
            raise AssertionError("full measurement has the wrong setting count")
        totals = [int(np.sum(counts)) for counts in result.counts.values()]
        if any(total != shots for total in totals):
            raise AssertionError("a measurement setting did not conserve shots")
        return {"settings": len(result.counts), "all_shot_totals_valid": True}

    estimate = result[0] if isinstance(result, tuple) else result
    hermitian = (estimate + estimate.conj().T) / 2.0
    diagnostics = {
        "trace_error": float(abs(np.trace(estimate) - 1.0)),
        "hermitian_error": float(np.linalg.norm(estimate - estimate.conj().T)),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(hermitian)[0]),
    }
    if diagnostics["trace_error"] > 2e-9 or diagnostics["hermitian_error"] > 2e-9:
        raise AssertionError(f"{stage} returned an invalid estimate: {diagnostics}")
    if stage == "fixed_iteration_mle":
        history = np.asarray(result[1], dtype=float)
        if diagnostics["minimum_eigenvalue"] < -2e-9:
            raise AssertionError(f"MLE estimate is not positive semidefinite: {diagnostics}")
        if np.any(np.diff(history) > 2e-11):
            raise AssertionError("MLE likelihood history is not monotone")
        diagnostics["accepted_iterations"] = int(len(history) - 1)
    return diagnostics


def execute_worker_case(args: argparse.Namespace) -> dict[str, Any]:
    """Run one isolated NumPy/qubit/stage case."""
    n_qubits = args.worker_qubits
    stage = args.worker_stage
    seed = args.seed + 1009 * n_qubits
    generator = np.random.default_rng(seed + 17)

    with PeakRssTracker() as memory:
        if stage == "state_generation":
            operation = lambda: haar_random_pure(n_qubits, xp=np, rng=seed)
        elif stage == "measurement_storage":
            operation = lambda: allocate_measurement_storage(n_qubits)
        else:
            rho = haar_random_pure(n_qubits, xp=np, rng=seed)
            if stage == "single_setting":
                setting = setting_for(n_qubits)
                operation = lambda: pauli_probabilities(rho, setting)
            elif stage == "full_measurement":
                operation = lambda: simulate_pauli_measurements(
                    rho, args.shots, rng=generator
                )
            else:
                data = simulate_pauli_measurements(rho, args.shots, rng=generator)
                if stage == "linear_inversion":
                    operation = lambda: linear_inversion_pauli(data)
                elif stage == "fixed_iteration_mle":
                    linear = linear_inversion_pauli(data)
                    initial = project_density_matrix(linear)
                    operation = lambda: factorized_mle(
                        data,
                        initial=initial,
                        rank=1,
                        max_iter=args.mle_iterations,
                        tolerance=0.0,
                        return_history=True,
                    )
                else:
                    raise ValueError(f"unsupported stage: {stage}")

        large_storage_case = stage == "measurement_storage" and n_qubits >= 9
        case_warmups = 0 if large_storage_case else args.warmups
        case_repeats = 1 if large_storage_case else args.repeats
        samples, result = timed_samples(operation, case_warmups, case_repeats)
        diagnostics = validate_result(stage, result, n_qubits, args.shots)

    expected_output_bytes = {
        "state_generation": 16 * 4**n_qubits,
        "single_setting": 8 * 2**n_qubits,
        "measurement_storage": 8 * 6**n_qubits,
        "full_measurement": 8 * 6**n_qubits,
        "linear_inversion": 16 * 4**n_qubits,
        "fixed_iteration_mle": 16 * 4**n_qubits,
    }[stage]
    output_bytes = native_nbytes(result)
    if output_bytes != expected_output_bytes:
        raise AssertionError(
            f"{stage} output uses {output_bytes} bytes, expected {expected_output_bytes}"
        )

    return {
        "n_qubits": n_qubits,
        "dimension": 2**n_qubits,
        "stage": stage,
        "settings": 3**n_qubits,
        "outcomes_per_setting": 2**n_qubits,
        "timing_samples_seconds": samples,
        "warmups": case_warmups,
        "timed_repeats": case_repeats,
        "median_seconds": float(np.median(samples)),
        "minimum_seconds": float(np.min(samples)),
        "baseline_rss_bytes": memory.baseline,
        "peak_rss_bytes": memory.peak,
        "peak_rss_delta_bytes": memory.peak - memory.baseline,
        "output_bytes": output_bytes,
        "expected_output_bytes": expected_output_bytes,
        "diagnostics": diagnostics,
    }


def worker_command(args: argparse.Namespace, n_qubits: int, stage: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-qubits",
        str(n_qubits),
        "--worker-stage",
        stage,
        "--shots",
        str(args.shots),
        "--repeats",
        str(args.repeats),
        "--warmups",
        str(args.warmups),
        "--mle-iterations",
        str(args.mle_iterations),
        "--seed",
        str(args.seed),
    ]


def run_isolated_case(
    args: argparse.Namespace, n_qubits: int, stage: str
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        worker_command(args, n_qubits, stage),
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.case_timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for n={n_qubits}, {stage}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result_lines = [
        line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise RuntimeError(f"worker returned no unique result: {completed.stdout}")
    return json.loads(result_lines[0][len(RESULT_PREFIX) :])


def collect_results(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    for n_qubits in range(1, args.max_qubits + 1):
        for stage in STAGES:
            if stage in ("full_measurement", "linear_inversion") and n_qubits > args.kernel_max_qubits:
                continue
            if stage == "fixed_iteration_mle" and n_qubits > args.mle_max_qubits:
                continue
            print(f"Running n={n_qubits} {STAGE_LABELS[stage]} ...", flush=True)
            record = run_isolated_case(args, n_qubits, stage)
            records.append(record)
            print(
                f"  median={record['median_seconds']:.4g} s, "
                f"peak delta={record['peak_rss_delta_bytes'] / MIB:.2f} MiB"
            )
    return records


def records_for(records: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in records if row["stage"] == stage],
        key=lambda row: row["n_qubits"],
    )


def scaling_slopes(records: list[dict[str, Any]]) -> dict[str, float]:
    slopes = {}
    for stage in STAGES:
        rows = records_for(records, stage)
        if len(rows) < 2:
            continue
        x = np.asarray([row["n_qubits"] for row in rows], dtype=float)
        y = np.asarray([max(row["median_seconds"], 1e-12) for row in rows])
        slopes[stage] = float(np.polyfit(x, np.log2(y), 1)[0])
    return slopes


def print_report(records: list[dict[str, Any]], slopes: dict[str, float]) -> None:
    print("\nNumPy qubit-count runtime and memory scaling verification")
    print("=" * 62)
    print(f"NumPy version: {np.__version__}; dtype: complex128; device: CPU")
    print("\nLargest measured case by stage")
    print(f"{'stage':<22} {'n':>2} {'median (s)':>12} {'peak delta':>12}")
    for stage in STAGES:
        row = records_for(records, stage)[-1]
        print(
            f"{STAGE_LABELS[stage]:<22} {row['n_qubits']:>2d} "
            f"{row['median_seconds']:>12.5g} "
            f"{row['peak_rss_delta_bytes'] / MIB:>9.2f} MiB"
        )
    print("\nFitted log2(time) slope per added qubit")
    for stage, slope in slopes.items():
        print(f"  {STAGE_LABELS[stage]:<25} {slope:7.3f}  (~{2**slope:.2f}x/qubit)")


def plot_results(
    records: list[dict[str, Any]],
    slopes: dict[str, float],
    output: Path,
    show: bool,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.6), constrained_layout=True)

    runtime_axis = axes[0, 0]
    for stage in STAGES:
        rows = records_for(records, stage)
        runtime_axis.semilogy(
            [row["n_qubits"] for row in rows],
            [row["median_seconds"] for row in rows],
            marker="o",
            color=STAGE_COLORS[stage],
            linewidth=1.8,
            label=STAGE_LABELS[stage],
        )
    runtime_axis.set_xlabel("number of qubits")
    runtime_axis.set_ylabel("median wall time (s)")
    runtime_axis.set_title("A. Runtime by tomography stage")
    runtime_axis.grid(which="both", alpha=0.23)
    runtime_axis.legend(fontsize=8)

    memory_axis = axes[0, 1]
    for stage in STAGES:
        rows = records_for(records, stage)
        memory_axis.semilogy(
            [row["n_qubits"] for row in rows],
            [max(row["peak_rss_delta_bytes"] / MIB, 1e-3) for row in rows],
            marker="o",
            color=STAGE_COLORS[stage],
            linewidth=1.8,
            label=STAGE_LABELS[stage],
        )
    memory_axis.set_xlabel("number of qubits")
    memory_axis.set_ylabel("peak RSS increase (MiB)")
    memory_axis.set_title("B. Isolated-process working-set growth")
    memory_axis.grid(which="both", alpha=0.23)
    memory_axis.legend(fontsize=8)

    growth_axis = axes[1, 0]
    stage_positions = np.arange(len(STAGES))
    multipliers = [2 ** slopes[stage] for stage in STAGES]
    bars = growth_axis.bar(
        stage_positions,
        multipliers,
        color=[STAGE_COLORS[stage] for stage in STAGES],
        width=0.68,
    )
    growth_axis.axhline(1.0, color="0.3", linestyle="--", linewidth=1.3)
    growth_axis.set_xticks(
        stage_positions,
        [STAGE_LABELS[stage].replace(" ", "\n") for stage in STAGES],
    )
    growth_axis.set_ylabel("fitted time multiplier per added qubit")
    growth_axis.set_title("C. Empirical runtime growth")
    growth_axis.grid(axis="y", alpha=0.23)
    growth_axis.bar_label(bars, fmt="%.2fx", padding=3, fontsize=9)

    storage_axis = axes[1, 1]
    qubits = np.arange(1, max(row["n_qubits"] for row in records) + 1)
    storage_axis.semilogy(
        qubits,
        16 * 4.0**qubits / MIB,
        marker="o",
        color="#3569a8",
        label=r"dense $\rho$: $16\,4^n$ bytes",
    )
    storage_axis.semilogy(
        qubits,
        8 * 6.0**qubits / MIB,
        marker="s",
        color="#cf6a32",
        label=r"counts: $8\,6^n$ bytes",
    )
    storage_axis.set_xlabel("number of qubits")
    storage_axis.set_ylabel("array payload (MiB)")
    storage_axis.set_title("D. Required persistent array storage")
    storage_axis.grid(which="both", alpha=0.23)
    storage_axis.legend(fontsize=9)

    figure.suptitle("Dense tomography qubit scaling with NumPy", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_results(
    path: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    slopes: dict[str, float],
) -> None:
    payload = {
        "metadata": {
            "implementation": "numpy",
            "numpy_version": np.__version__,
            "dtype": "complex128",
            "device": "cpu",
            "max_qubits": args.max_qubits,
            "kernel_max_qubits": args.kernel_max_qubits,
            "mle_max_qubits": args.mle_max_qubits,
            "shots_per_setting": args.shots,
            "warmups": args.warmups,
            "timed_repeats": args.repeats,
            "mle_iterations": args.mle_iterations,
            "seed": args.seed,
            "memory_method": "fresh subprocess peak RSS minus process baseline",
        },
        "summary": {"log2_time_slopes": slopes},
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-qubits", type=int, default=DEFAULT_MAX_QUBITS)
    parser.add_argument("--kernel-max-qubits", type=int, default=DEFAULT_KERNEL_MAX_QUBITS)
    parser.add_argument("--mle-max-qubits", type=int, default=DEFAULT_MLE_MAX_QUBITS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--mle-iterations", type=int, default=DEFAULT_MLE_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--case-timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-qubits", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_qubits < 1 or args.kernel_max_qubits < 1 or args.mle_max_qubits < 1:
        parser.error("qubit limits must be positive")
    if args.repeats < 1 or args.warmups < 0 or args.mle_iterations < 1:
        parser.error(
            "repeats and MLE iterations must be positive; warmups cannot be negative"
        )
    if args.worker and None in (args.worker_qubits, args.worker_stage):
        parser.error("worker mode requires qubits and stage")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        result = execute_worker_case(args)
        print(RESULT_PREFIX + json.dumps(result))
        return

    records = collect_results(args)
    slopes = scaling_slopes(records)
    print_report(records, slopes)
    plot_results(records, slopes, args.output, args.show)
    save_results(args.json, args, records, slopes)
    print(f"\nSaved figure: {args.output.resolve()}")
    print(f"Saved data:   {args.json.resolve()}")


if __name__ == "__main__":
    main()
