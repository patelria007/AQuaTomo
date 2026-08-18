"""Verify NumPy reconstruction quality as the qubit count grows to six.

Run from the repository root with::

    python verification/5_qubit_scaling/reconstruction_quality_scaling_verification.py

The experiment separates exact-probability algebraic closure from two
finite-shot resource models: fixed shots per setting and a fixed total shot
budget shared by all complete local-Pauli settings.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import psutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nbqst.denoise import project_density_matrix  # noqa: E402
from nbqst.measurements import (  # noqa: E402
    exact_pauli_measurements,
    simulate_pauli_measurements,
)
from nbqst.metrics import (  # noqa: E402
    fidelity,
    hilbert_schmidt_distance,
    purity,
    trace_distance,
)
from nbqst.reconstruction import linear_inversion_pauli  # noqa: E402
from nbqst.states import (  # noqa: E402
    ghz_state,
    haar_random_pure,
    random_mixed_state,
    random_product_state,
)


MAX_QUBITS = 6
FIXED_SHOTS_PER_SETTING = (64, 256, 1024)
DEFAULT_TRIALS = 4
DEFAULT_SEED = 20261021
TOTAL_SHOT_BUDGET = 256 * 3**MAX_QUBITS
EXACT_TOLERANCE = 1e-10
PHYSICAL_TOLERANCE = 2e-9
DEFAULT_OUTPUT = Path(__file__).with_name(
    "reconstruction_quality_scaling_verification.png"
)
DEFAULT_JSON = Path(__file__).with_name(
    "reconstruction_quality_scaling_results.json"
)
MIB = 1024.0**2


@dataclass(frozen=True)
class StateFamily:
    name: str
    build: Callable[[int, int], np.ndarray]
    minimum_qubits: int = 1


STATE_FAMILIES = (
    StateFamily(
        "Product pure",
        lambda n_qubits, seed: np.asarray(
            random_product_state(n_qubits, rng=seed)
        ),
    ),
    StateFamily(
        "Haar pure",
        lambda n_qubits, seed: np.asarray(
            haar_random_pure(n_qubits, rng=seed)
        ),
    ),
    StateFamily(
        "GHZ",
        lambda n_qubits, seed: np.asarray(ghz_state(n_qubits)),
        minimum_qubits=2,
    ),
    StateFamily(
        "Rank-4 mixed",
        lambda n_qubits, seed: np.asarray(
            random_mixed_state(
                n_qubits,
                rank=min(4, 2**n_qubits),
                rng=seed,
            )
        ),
    ),
)
FINITE_FAMILIES = tuple(
    family for family in STATE_FAMILIES if family.name in ("Haar pure", "Rank-4 mixed")
)


class PeakRssTracker:
    """Poll process RSS throughout the complete quality experiment."""

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


def state_seed(seed: int, family_index: int, n_qubits: int) -> int:
    sequence = np.random.SeedSequence([seed, 17, family_index, n_qubits])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def measurement_seed(
    seed: int,
    protocol_index: int,
    family_index: int,
    n_qubits: int,
    shots_per_setting: int,
    trial: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            seed,
            31,
            protocol_index,
            family_index,
            n_qubits,
            shots_per_setting,
            trial,
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def density_diagnostics(rho: np.ndarray) -> dict[str, float]:
    rho = np.asarray(rho)
    hermitian = (rho + rho.conj().T) / 2.0
    return {
        "trace_error": float(abs(np.trace(rho) - 1.0)),
        "hermitian_error": float(np.linalg.norm(rho - rho.conj().T)),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(hermitian)[0]),
    }


def assert_physical(name: str, rho: np.ndarray) -> dict[str, float]:
    diagnostics = density_diagnostics(rho)
    if diagnostics["trace_error"] > PHYSICAL_TOLERANCE:
        raise AssertionError(f"{name} does not have unit trace: {diagnostics}")
    if diagnostics["hermitian_error"] > PHYSICAL_TOLERANCE:
        raise AssertionError(f"{name} is not Hermitian: {diagnostics}")
    if diagnostics["minimum_eigenvalue"] < -PHYSICAL_TOLERANCE:
        raise AssertionError(f"{name} is not positive semidefinite: {diagnostics}")
    return diagnostics


def measurement_payload_bytes(data) -> int:
    return sum(int(np.asarray(counts).nbytes) for counts in data.counts.values())


def quality_metrics(
    rho_true: np.ndarray,
    linear: np.ndarray,
    projected: np.ndarray,
) -> dict[str, float]:
    linear_diagnostics = density_diagnostics(linear)
    projected_diagnostics = assert_physical("projected linear estimate", projected)
    projected_fidelity = float(fidelity(rho_true, projected))
    if not -1e-9 <= projected_fidelity <= 1.0 + 1e-8:
        raise AssertionError(f"projected fidelity is invalid: {projected_fidelity}")
    projected_fidelity = float(np.clip(projected_fidelity, 0.0, 1.0))
    return {
        "linear_hs_distance": float(hilbert_schmidt_distance(rho_true, linear)),
        "projected_hs_distance": float(
            hilbert_schmidt_distance(rho_true, projected)
        ),
        "projected_trace_distance": float(trace_distance(rho_true, projected)),
        "projected_fidelity": projected_fidelity,
        "projected_infidelity": 1.0 - projected_fidelity,
        "target_purity": float(purity(rho_true)),
        "projected_purity": float(purity(projected)),
        "linear_minimum_eigenvalue": linear_diagnostics["minimum_eigenvalue"],
        "linear_is_nonphysical": bool(
            linear_diagnostics["minimum_eigenvalue"] < -PHYSICAL_TOLERANCE
        ),
        "projected_minimum_eigenvalue": projected_diagnostics[
            "minimum_eigenvalue"
        ],
    }


def run_exact_closure(seed: int) -> list[dict[str, object]]:
    records = []
    for n_qubits in range(1, MAX_QUBITS + 1):
        for family_index, family in enumerate(STATE_FAMILIES):
            if n_qubits < family.minimum_qubits:
                continue
            rho_true = family.build(
                n_qubits,
                state_seed(seed, family_index, n_qubits),
            )
            assert_physical(f"{family.name} target", rho_true)

            start = time.perf_counter()
            data = exact_pauli_measurements(rho_true)
            measurement_seconds = time.perf_counter() - start
            start = time.perf_counter()
            estimate = np.asarray(linear_inversion_pauli(data))
            reconstruction_seconds = time.perf_counter() - start
            error = float(hilbert_schmidt_distance(rho_true, estimate))
            if error > EXACT_TOLERANCE:
                raise AssertionError(
                    f"exact closure failed for {family.name}, n={n_qubits}: {error:.3e}"
                )
            diagnostics = density_diagnostics(estimate)
            records.append(
                {
                    "family": family.name,
                    "n_qubits": n_qubits,
                    "hs_distance": error,
                    "measurement_seconds": measurement_seconds,
                    "reconstruction_seconds": reconstruction_seconds,
                    "live_array_payload_bytes": int(
                        rho_true.nbytes
                        + measurement_payload_bytes(data)
                        + estimate.nbytes
                    ),
                    **diagnostics,
                }
            )
        gc.collect()
    return records


def run_finite_case(
    *,
    protocol: str,
    protocol_index: int,
    family: StateFamily,
    family_index: int,
    rho_true: np.ndarray,
    n_qubits: int,
    shots_per_setting: int,
    trial: int,
    seed: int,
) -> dict[str, object]:
    case_start = time.perf_counter()
    start = time.perf_counter()
    data = simulate_pauli_measurements(
        rho_true,
        shots_per_setting,
        rng=measurement_seed(
            seed,
            protocol_index,
            family_index,
            n_qubits,
            shots_per_setting,
            trial,
        ),
    )
    measurement_seconds = time.perf_counter() - start
    if len(data.counts) != 3**n_qubits:
        raise AssertionError("finite-shot data have the wrong setting count")
    if any(int(np.sum(counts)) != shots_per_setting for counts in data.counts.values()):
        raise AssertionError("a finite-shot setting did not conserve shots")

    start = time.perf_counter()
    linear = np.asarray(linear_inversion_pauli(data))
    linear_seconds = time.perf_counter() - start
    start = time.perf_counter()
    projected = np.asarray(project_density_matrix(linear))
    projection_seconds = time.perf_counter() - start
    start = time.perf_counter()
    metrics = quality_metrics(rho_true, linear, projected)
    metrics_seconds = time.perf_counter() - start

    return {
        "protocol": protocol,
        "family": family.name,
        "n_qubits": n_qubits,
        "trial": trial,
        "shots_per_setting": shots_per_setting,
        "settings": 3**n_qubits,
        "total_shots": shots_per_setting * 3**n_qubits,
        "measurement_seconds": measurement_seconds,
        "linear_seconds": linear_seconds,
        "projection_seconds": projection_seconds,
        "metrics_seconds": metrics_seconds,
        "case_seconds": time.perf_counter() - case_start,
        "live_array_payload_bytes": int(
            rho_true.nbytes
            + measurement_payload_bytes(data)
            + linear.nbytes
            + projected.nbytes
        ),
        **metrics,
    }


def run_finite_trials(trials: int, seed: int) -> list[dict[str, object]]:
    if trials < 3:
        raise ValueError("trials must be at least 3")
    records = []
    family_indices = {family.name: STATE_FAMILIES.index(family) for family in FINITE_FAMILIES}

    for n_qubits in range(1, MAX_QUBITS + 1):
        for finite_index, family in enumerate(FINITE_FAMILIES):
            family_index = family_indices[family.name]
            rho_true = family.build(
                n_qubits,
                state_seed(seed, family_index, n_qubits),
            )
            assert_physical(f"{family.name} target", rho_true)

            for shots_per_setting in FIXED_SHOTS_PER_SETTING:
                for trial in range(trials):
                    records.append(
                        run_finite_case(
                            protocol="fixed_shots_per_setting",
                            protocol_index=0,
                            family=family,
                            family_index=finite_index,
                            rho_true=rho_true,
                            n_qubits=n_qubits,
                            shots_per_setting=shots_per_setting,
                            trial=trial,
                            seed=seed,
                        )
                    )

            budget_shots_per_setting = TOTAL_SHOT_BUDGET // 3**n_qubits
            if budget_shots_per_setting * 3**n_qubits != TOTAL_SHOT_BUDGET:
                raise AssertionError("total shot budget is not divisible by the setting count")
            for trial in range(trials):
                records.append(
                    run_finite_case(
                        protocol="fixed_total_shots",
                        protocol_index=1,
                        family=family,
                        family_index=finite_index,
                        rho_true=rho_true,
                        n_qubits=n_qubits,
                        shots_per_setting=budget_shots_per_setting,
                        trial=trial,
                        seed=seed,
                    )
                )
        gc.collect()
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in records:
        key = (
            row["protocol"],
            row["family"],
            row["n_qubits"],
            row["shots_per_setting"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, rows in sorted(grouped.items()):
        infidelities = np.asarray([row["projected_infidelity"] for row in rows])
        linear_hs = np.asarray([row["linear_hs_distance"] for row in rows])
        projected_hs = np.asarray([row["projected_hs_distance"] for row in rows])
        summary.append(
            {
                "protocol": key[0],
                "family": key[1],
                "n_qubits": key[2],
                "shots_per_setting": key[3],
                "total_shots": rows[0]["total_shots"],
                "trials": len(rows),
                "mean_projected_infidelity": float(np.mean(infidelities)),
                "std_projected_infidelity": float(np.std(infidelities, ddof=1)),
                "rms_linear_hs_distance": float(np.sqrt(np.mean(linear_hs**2))),
                "rms_projected_hs_distance": float(np.sqrt(np.mean(projected_hs**2))),
                "linear_nonphysical_fraction": float(
                    np.mean([row["linear_is_nonphysical"] for row in rows])
                ),
                "mean_projected_purity": float(
                    np.mean([row["projected_purity"] for row in rows])
                ),
                "median_case_seconds": float(
                    np.median([row["case_seconds"] for row in rows])
                ),
                "mean_live_array_payload_bytes": float(
                    np.mean([row["live_array_payload_bytes"] for row in rows])
                ),
            }
        )
    return summary


def summary_rows(
    summary: list[dict[str, object]],
    *,
    protocol: str,
    family: str | None = None,
    shots_per_setting: int | None = None,
) -> list[dict[str, object]]:
    rows = [row for row in summary if row["protocol"] == protocol]
    if family is not None:
        rows = [row for row in rows if row["family"] == family]
    if shots_per_setting is not None:
        rows = [row for row in rows if row["shots_per_setting"] == shots_per_setting]
    return sorted(rows, key=lambda row: (row["n_qubits"], row["family"]))


def print_report(
    exact_records: list[dict[str, object]],
    summary: list[dict[str, object]],
    trials: int,
    elapsed_seconds: float,
    memory: PeakRssTracker,
) -> None:
    print("\nNumPy reconstruction-quality scaling verification")
    print("=" * 57)
    print(f"Maximum exact-closure HS distance: {max(row['hs_distance'] for row in exact_records):.3e}")
    print(f"Finite-shot trials per condition:  {trials}")
    print(f"Total experiment wall time:        {elapsed_seconds:.2f} s")
    print(f"Peak RSS increase:                 {(memory.peak - memory.baseline) / MIB:.2f} MiB")
    print("\nSix-qubit projected linear-inversion quality")
    print(
        f"{'protocol':<23} {'family':<14} {'shots/set':>9} "
        f"{'total shots':>12} {'mean 1-F':>11} {'RMS HS':>10} {'case (s)':>9}"
    )
    for row in summary:
        if row["n_qubits"] != MAX_QUBITS:
            continue
        protocol = (
            "fixed shots/setting"
            if row["protocol"] == "fixed_shots_per_setting"
            else "fixed total shots"
        )
        print(
            f"{protocol:<23} {row['family']:<14} {row['shots_per_setting']:>9d} "
            f"{row['total_shots']:>12d} {row['mean_projected_infidelity']:>11.4g} "
            f"{row['rms_projected_hs_distance']:>10.4g} "
            f"{row['median_case_seconds']:>9.3f}"
        )


def plot_results(
    exact_records: list[dict[str, object]],
    summary: list[dict[str, object]],
    output: Path,
    trials: int,
    show: bool,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.7), constrained_layout=True)
    family_colors = {"Haar pure": "#3569a8", "Rank-4 mixed": "#cf6a32"}
    shot_markers = {64: "o", 256: "s", 1024: "^"}

    exact_axis = axes[0, 0]
    exact_colors = {
        "Product pure": "#3569a8",
        "Haar pure": "#6b8fd3",
        "GHZ": "#3a8b5b",
        "Rank-4 mixed": "#cf6a32",
    }
    for family in STATE_FAMILIES:
        rows = sorted(
            [row for row in exact_records if row["family"] == family.name],
            key=lambda row: row["n_qubits"],
        )
        exact_axis.semilogy(
            [row["n_qubits"] for row in rows],
            [max(row["hs_distance"], 1e-17) for row in rows],
            marker="o",
            color=exact_colors[family.name],
            label=family.name,
        )
    exact_axis.axhline(
        EXACT_TOLERANCE,
        color="0.3",
        linestyle="--",
        linewidth=1.3,
        label="acceptance threshold",
    )
    exact_axis.set_xlabel("number of qubits")
    exact_axis.set_ylabel("exact round-trip HS distance")
    exact_axis.set_title("A. Exact probabilities close the pipeline")
    exact_axis.grid(which="both", alpha=0.23)
    exact_axis.legend(fontsize=8)

    fixed_axis = axes[0, 1]
    for family in FINITE_FAMILIES:
        for shots in FIXED_SHOTS_PER_SETTING:
            rows = summary_rows(
                summary,
                protocol="fixed_shots_per_setting",
                family=family.name,
                shots_per_setting=shots,
            )
            fixed_axis.semilogy(
                [row["n_qubits"] for row in rows],
                [max(row["mean_projected_infidelity"], 1e-8) for row in rows],
                marker=shot_markers[shots],
                color=family_colors[family.name],
                linestyle="-" if family.name == "Haar pure" else "--",
                label=f"{family.name}, {shots}/setting",
            )
    fixed_axis.set_xlabel("number of qubits")
    fixed_axis.set_ylabel(r"mean projected infidelity $1-F$")
    fixed_axis.set_title("B. Fixed shots per setting")
    fixed_axis.grid(which="both", alpha=0.23)
    fixed_axis.legend(fontsize=7)

    budget_axis = axes[0, 2]
    for family in FINITE_FAMILIES:
        rows = summary_rows(
            summary,
            protocol="fixed_total_shots",
            family=family.name,
        )
        budget_axis.semilogy(
            [row["n_qubits"] for row in rows],
            [max(row["mean_projected_infidelity"], 1e-8) for row in rows],
            marker="o",
            color=family_colors[family.name],
            label=family.name,
        )
    budget_axis.set_xlabel("number of qubits")
    budget_axis.set_ylabel(r"mean projected infidelity $1-F$")
    budget_axis.set_title(f"C. Fixed total budget ({TOTAL_SHOT_BUDGET:,} shots)")
    budget_axis.grid(which="both", alpha=0.23)
    budget_axis.legend(fontsize=8)

    physicality_axis = axes[1, 0]
    for family in FINITE_FAMILIES:
        for shots in FIXED_SHOTS_PER_SETTING:
            rows = summary_rows(
                summary,
                protocol="fixed_shots_per_setting",
                family=family.name,
                shots_per_setting=shots,
            )
            physicality_axis.plot(
                [row["n_qubits"] for row in rows],
                [row["linear_nonphysical_fraction"] for row in rows],
                marker=shot_markers[shots],
                color=family_colors[family.name],
                linestyle="-" if family.name == "Haar pure" else "--",
                label=f"{family.name}, {shots}/setting",
            )
    physicality_axis.set_ylim(-0.04, 1.04)
    physicality_axis.set_xlabel("number of qubits")
    physicality_axis.set_ylabel("raw LI nonphysical fraction")
    physicality_axis.set_title("D. Projection is increasingly necessary")
    physicality_axis.grid(alpha=0.23)
    physicality_axis.legend(fontsize=7)

    runtime_axis = axes[1, 1]
    for protocol, label, marker in (
        ("fixed_shots_per_setting", "fixed shots/setting", "o"),
        ("fixed_total_shots", "fixed total shots", "s"),
    ):
        medians = []
        for n_qubits in range(1, MAX_QUBITS + 1):
            rows = [
                row
                for row in summary
                if row["protocol"] == protocol and row["n_qubits"] == n_qubits
            ]
            medians.append(float(np.median([row["median_case_seconds"] for row in rows])))
        runtime_axis.semilogy(
            range(1, MAX_QUBITS + 1),
            medians,
            marker=marker,
            linewidth=2,
            label=label,
        )
    runtime_axis.set_xlabel("number of qubits")
    runtime_axis.set_ylabel("median finite-shot case time (s)")
    runtime_axis.set_title("E. Measurement + LI + projection + metrics")
    runtime_axis.grid(which="both", alpha=0.23)
    runtime_axis.legend(fontsize=8)

    payload_axis = axes[1, 2]
    payloads = []
    for n_qubits in range(1, MAX_QUBITS + 1):
        rows = [row for row in summary if row["n_qubits"] == n_qubits]
        payloads.append(float(np.mean([row["mean_live_array_payload_bytes"] for row in rows])) / MIB)
    payload_axis.semilogy(
        range(1, MAX_QUBITS + 1),
        payloads,
        marker="o",
        color="#9b59b6",
        linewidth=2,
        label="target + counts + LI + projected",
    )
    payload_axis.set_xlabel("number of qubits")
    payload_axis.set_ylabel("live array payload (MiB)")
    payload_axis.set_title("F. Persistent arrays for one quality case")
    payload_axis.grid(which="both", alpha=0.23)
    payload_axis.legend(fontsize=8)

    figure.suptitle(
        f"Reconstruction-quality scaling through six qubits ({trials} trials/condition)",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_results(
    path: Path,
    args: argparse.Namespace,
    exact_records: list[dict[str, object]],
    finite_records: list[dict[str, object]],
    summary: list[dict[str, object]],
    elapsed_seconds: float,
    memory: PeakRssTracker,
) -> None:
    payload = {
        "metadata": {
            "implementation": "numpy",
            "numpy_version": np.__version__,
            "max_qubits": MAX_QUBITS,
            "finite_families": [family.name for family in FINITE_FAMILIES],
            "exact_families": [family.name for family in STATE_FAMILIES],
            "fixed_shots_per_setting": list(FIXED_SHOTS_PER_SETTING),
            "fixed_total_shot_budget": TOTAL_SHOT_BUDGET,
            "trials_per_condition": args.trials,
            "seed": args.seed,
            "estimator": "linear inversion followed by PSD unit-trace projection",
        },
        "resource_summary": {
            "elapsed_seconds": elapsed_seconds,
            "baseline_rss_bytes": memory.baseline,
            "peak_rss_bytes": memory.peak,
            "peak_rss_delta_bytes": memory.peak - memory.baseline,
        },
        "summary": {
            "maximum_exact_hs_distance": max(
                row["hs_distance"] for row in exact_records
            ),
            "conditions": summary,
        },
        "exact_records": exact_records,
        "finite_shot_records": finite_records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_start = time.perf_counter()
    with PeakRssTracker() as memory:
        exact_records = run_exact_closure(args.seed)
        finite_records = run_finite_trials(args.trials, args.seed)
        summary = summarize(finite_records)
    elapsed_seconds = time.perf_counter() - experiment_start
    print_report(exact_records, summary, args.trials, elapsed_seconds, memory)
    plot_results(exact_records, summary, args.output, args.trials, args.show)
    save_results(
        args.json,
        args,
        exact_records,
        finite_records,
        summary,
        elapsed_seconds,
        memory,
    )
    print(f"\nSaved figure: {args.output.resolve()}")
    print(f"Saved data:   {args.json.resolve()}")


if __name__ == "__main__":
    main()
