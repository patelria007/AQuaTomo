"""Verify the complete state -> measurement -> reconstruction pipeline.

Run from the repository root with::

    python verification/4_end_to_end_pipeline/end_to_end_verification.py

The verification has two complementary parts.  Deterministic exact-probability
data check algebraic closure across one to four qubits.  Repeated four-qubit
finite-shot experiments check statistical improvement, physicality, and MLE
likelihood monotonicity using only public package functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nbqst.denoise import project_density_matrix  # noqa: E402
from nbqst.measurements import (  # noqa: E402
    complete_pauli_settings,
    exact_pauli_measurements,
    simulate_pauli_measurements,
)
from nbqst.metrics import fidelity, hilbert_schmidt_distance  # noqa: E402
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli  # noqa: E402
from nbqst.states import (  # noqa: E402
    ghz_state,
    haar_random_pure,
    random_mixed_state,
    random_product_state,
)


DEFAULT_TRIALS = 8
DEFAULT_SEED = 20260920
DEFAULT_OUTPUT = Path(__file__).with_name("end_to_end_verification.png")
DEFAULT_JSON = Path(__file__).with_name("end_to_end_results.json")
SHOT_COUNTS = (64, 256, 1024, 4096)
N_QUBITS = 4
MLE_MAX_ITERATIONS = 50
PHYSICAL_TOLERANCE = 1e-10
EXACT_TOLERANCE = 5e-12


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
        "Rank-controlled mixed",
        lambda n_qubits, seed: np.asarray(
            random_mixed_state(n_qubits, rank=min(4, 2**n_qubits), rng=seed)
        ),
    ),
)


def density_diagnostics(rho: np.ndarray) -> dict[str, float]:
    """Return the physical density-matrix diagnostics used by this script."""
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


def state_seed(seed: int, family_index: int, n_qubits: int) -> int:
    """Keep target-state seeds independent of finite-shot sampling seeds."""
    sequence = np.random.SeedSequence([seed, 11, family_index, n_qubits])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def measurement_rng(
    seed: int,
    family_index: int,
    shots: int,
    trial: int,
) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, 29, family_index, shots, trial])
    return np.random.default_rng(sequence)


def run_exact_closure(seed: int) -> list[dict[str, object]]:
    """Check exact state recovery for every supported family at n=1,...,4."""
    records: list[dict[str, object]] = []
    for n_qubits in range(1, N_QUBITS + 1):
        for family_index, family in enumerate(STATE_FAMILIES):
            if n_qubits < family.minimum_qubits:
                continue
            rho_true = family.build(
                n_qubits,
                state_seed(seed, family_index, n_qubits),
            )
            assert_physical(f"{family.name} target at n={n_qubits}", rho_true)
            data = exact_pauli_measurements(rho_true)
            if not data.informationally_complete:
                raise AssertionError("exact measurement data are not informationally complete")
            estimate = np.asarray(linear_inversion_pauli(data))
            error = float(hilbert_schmidt_distance(rho_true, estimate))
            if error > EXACT_TOLERANCE:
                raise AssertionError(
                    f"exact closure failed for {family.name}, n={n_qubits}: {error:.3e}"
                )
            records.append(
                {
                    "family": family.name,
                    "n_qubits": n_qubits,
                    "hs_distance": error,
                    **density_diagnostics(estimate),
                }
            )
    return records


def run_finite_shot_trials(
    trials: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run repeated four-qubit end-to-end reconstructions."""
    if trials < 4:
        raise ValueError("trials must be at least 4 for aggregate checks")

    records: list[dict[str, object]] = []
    representative: dict[str, object] | None = None
    expected_settings = complete_pauli_settings(N_QUBITS)

    for family_index, family in enumerate(STATE_FAMILIES):
        rho_true = family.build(
            N_QUBITS,
            state_seed(seed, family_index, N_QUBITS),
        )
        assert_physical(f"{family.name} four-qubit target", rho_true)

        for shots in SHOT_COUNTS:
            for trial in range(trials):
                data = simulate_pauli_measurements(
                    rho_true,
                    shots,
                    rng=measurement_rng(seed, family_index, shots, trial),
                )
                if data.settings != expected_settings or not data.informationally_complete:
                    raise AssertionError("finite-shot data lost a Pauli setting")
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
                history = np.asarray(history, dtype=float)

                linear_diagnostics = density_diagnostics(linear)
                if linear_diagnostics["trace_error"] > PHYSICAL_TOLERANCE:
                    raise AssertionError("linear inversion lost unit trace")
                if linear_diagnostics["hermitian_error"] > PHYSICAL_TOLERANCE:
                    raise AssertionError("linear inversion lost Hermiticity")
                assert_physical("projected estimate", projected)
                assert_physical("MLE estimate", mle)
                if np.any(np.diff(history) > 2e-12):
                    raise AssertionError("MLE accepted a likelihood-increasing step")

                estimates = {
                    "Linear": linear,
                    "Projected": projected,
                    "MLE": mle,
                }
                for method, estimate in estimates.items():
                    diagnostics = density_diagnostics(estimate)
                    row: dict[str, object] = {
                        "family": family.name,
                        "shots": shots,
                        "trial": trial,
                        "method": method,
                        "hs_distance": float(
                            hilbert_schmidt_distance(rho_true, estimate)
                        ),
                        **diagnostics,
                    }
                    if method != "Linear":
                        row["infidelity"] = max(
                            0.0,
                            1.0 - float(fidelity(rho_true, estimate)),
                        )
                    records.append(row)

                if family.name == "GHZ" and shots == 1024 and trial == 0:
                    representative = {
                        "rho_true": rho_true,
                        "frequencies": np.stack(
                            [
                                np.asarray(data.counts[setting], dtype=float) / shots
                                for setting in data.settings
                            ]
                        ),
                        "settings": data.settings,
                        "rho_mle": mle,
                        "history": history,
                        "fidelity": float(fidelity(rho_true, mle)),
                        "shots": shots,
                    }

    if representative is None:
        raise AssertionError("representative four-qubit pipeline was not collected")
    return records, representative


def rows_for(
    records: list[dict[str, object]],
    *,
    family: str | None = None,
    shots: int | None = None,
    method: str | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in records
        if (family is None or row["family"] == family)
        and (shots is None or row["shots"] == shots)
        and (method is None or row["method"] == method)
    ]


def analyze(
    exact_records: list[dict[str, object]],
    finite_records: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate the numerical assertions and plotting summaries."""
    max_exact_error = max(float(row["hs_distance"]) for row in exact_records)
    mean_infidelity: dict[tuple[str, str], np.ndarray] = {}
    nonphysical_fraction: dict[str, np.ndarray] = {}

    for family in STATE_FAMILIES:
        for method in ("Projected", "MLE"):
            mean_infidelity[(family.name, method)] = np.asarray(
                [
                    np.mean(
                        [
                            float(row["infidelity"])
                            for row in rows_for(
                                finite_records,
                                family=family.name,
                                shots=shots,
                                method=method,
                            )
                        ]
                    )
                    for shots in SHOT_COUNTS
                ]
            )

    for method in ("Linear", "Projected", "MLE"):
        nonphysical_fraction[method] = np.asarray(
            [
                np.mean(
                    [
                        float(row["minimum_eigenvalue"]) < -PHYSICAL_TOLERANCE
                        for row in rows_for(
                            finite_records,
                            shots=shots,
                            method=method,
                        )
                    ]
                )
                for shots in SHOT_COUNTS
            ]
        )

    for method in ("Projected", "MLE"):
        first = np.mean(
            [mean_infidelity[(family.name, method)][0] for family in STATE_FAMILIES]
        )
        last = np.mean(
            [mean_infidelity[(family.name, method)][-1] for family in STATE_FAMILIES]
        )
        if last >= first:
            raise AssertionError(
                f"aggregate {method} infidelity did not improve with shot count"
            )
    if np.any(nonphysical_fraction["Projected"] > 0.0):
        raise AssertionError("a projected estimate was nonphysical")
    if np.any(nonphysical_fraction["MLE"] > 0.0):
        raise AssertionError("an MLE estimate was nonphysical")

    return {
        "max_exact_error": max_exact_error,
        "mean_infidelity": mean_infidelity,
        "nonphysical_fraction": nonphysical_fraction,
    }


def print_report(
    exact_records: list[dict[str, object]],
    finite_records: list[dict[str, object]],
    analysis: dict[str, object],
    representative: dict[str, object],
    trials: int,
) -> None:
    print("Four-qubit end-to-end pipeline verification")
    print(f"Exact closure cases: {len(exact_records)}")
    print(f"Maximum exact HS distance: {analysis['max_exact_error']:.3e}")
    print(f"Finite-shot reconstructions: {len(finite_records)}")
    print(f"Trials per family/shot condition: {trials}")
    print(
        "Representative GHZ MLE fidelity "
        f"({representative['shots']} shots/setting): "
        f"{representative['fidelity']:.6f}"
    )
    print("\nMean MLE infidelity at 64 -> 4096 shots/setting")
    for family in STATE_FAMILIES:
        values = analysis["mean_infidelity"][(family.name, "MLE")]
        print(f"  {family.name:<22} {values[0]:.4e} -> {values[-1]:.4e}")
    print("\nNonphysical fraction at 64 -> 4096 shots/setting")
    for method in ("Linear", "Projected", "MLE"):
        values = analysis["nonphysical_fraction"][method]
        print(f"  {method:<10} {values[0]:.3f} -> {values[-1]:.3f}")


def plot_results(
    exact_records: list[dict[str, object]],
    analysis: dict[str, object],
    representative: dict[str, object],
    output: Path,
    trials: int,
    *,
    show: bool = False,
) -> None:
    """Render the complete pipeline and its aggregate validation evidence."""
    family_colors = {
        family.name: color
        for family, color in zip(
            STATE_FAMILIES,
            ("#3366a8", "#8d55b5", "#2f8b57", "#ce682f"),
        )
    }
    method_colors = {
        "Linear": "#3366a8",
        "Projected": "#ce682f",
        "MLE": "#2f8b57",
    }

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), layout="constrained")
    figure.suptitle(
        f"Four-qubit end-to-end tomography verification ({trials} trials per condition)",
        fontsize=20,
    )

    true_axis, frequency_axis, estimate_axis = axes[0]
    matrix_vmax = max(
        float(np.max(np.abs(representative["rho_true"]))),
        float(np.max(np.abs(representative["rho_mle"]))),
    )
    true_image = true_axis.imshow(
        np.abs(representative["rho_true"]),
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=matrix_vmax,
        aspect="equal",
    )
    true_axis.set_title("A. Generated four-qubit GHZ state")
    true_axis.set_xlabel("computational-basis column")
    true_axis.set_ylabel("computational-basis row")
    figure.colorbar(true_image, ax=true_axis, label=r"$|\rho_{ij}|$")

    frequencies = np.asarray(representative["frequencies"])
    frequency_image = frequency_axis.imshow(
        frequencies,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.max(frequencies)),
        aspect="auto",
    )
    setting_ticks = np.linspace(0, len(representative["settings"]) - 1, 5, dtype=int)
    frequency_axis.set_yticks(setting_ticks)
    frequency_axis.set_yticklabels(
        [representative["settings"][index] for index in setting_ticks]
    )
    outcome_ticks = np.asarray([0, 3, 7, 11, 15])
    frequency_axis.set_xticks(outcome_ticks)
    frequency_axis.set_xticklabels([f"{value:04b}" for value in outcome_ticks])
    frequency_axis.set_title(
        f"B. Complete Pauli data ({representative['shots']} shots/setting)"
    )
    frequency_axis.set_xlabel("measurement outcome")
    frequency_axis.set_ylabel("Pauli setting (81 total)")
    figure.colorbar(frequency_image, ax=frequency_axis, label="observed frequency")

    estimate_image = estimate_axis.imshow(
        np.abs(representative["rho_mle"]),
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=matrix_vmax,
        aspect="equal",
    )
    estimate_axis.set_title(
        "C. Physical MLE reconstruction\n"
        f"fidelity={representative['fidelity']:.5f}"
    )
    estimate_axis.set_xlabel("computational-basis column")
    estimate_axis.set_ylabel("computational-basis row")
    figure.colorbar(estimate_image, ax=estimate_axis, label=r"$|\hat{\rho}_{ij}|$")

    exact_axis = axes[1, 0]
    for family in STATE_FAMILIES:
        rows = [row for row in exact_records if row["family"] == family.name]
        exact_axis.semilogy(
            [row["n_qubits"] for row in rows],
            [max(float(row["hs_distance"]), 1e-18) for row in rows],
            marker="o",
            linewidth=2,
            color=family_colors[family.name],
            label=family.name,
        )
    exact_axis.axhline(
        EXACT_TOLERANCE,
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        label=f"acceptance threshold ({EXACT_TOLERANCE:.0e})",
    )
    exact_axis.set_xticks(range(1, N_QUBITS + 1))
    exact_axis.set_xlabel("number of qubits")
    exact_axis.set_ylabel(r"exact round-trip HS distance")
    exact_axis.set_title("D. Exact probabilities close the pipeline")
    exact_axis.grid(which="both", alpha=0.25)
    exact_axis.legend(fontsize=8)

    fidelity_axis = axes[1, 1]
    for family in STATE_FAMILIES:
        color = family_colors[family.name]
        for method, linestyle, marker in (
            ("Projected", "--", "s"),
            ("MLE", "-", "o"),
        ):
            values = analysis["mean_infidelity"][(family.name, method)]
            fidelity_axis.loglog(
                SHOT_COUNTS,
                np.maximum(values, 1e-8),
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                label=f"{family.name}, {method}",
            )
    fidelity_axis.set_xlabel("shots per Pauli setting")
    fidelity_axis.set_ylabel(r"mean physical-estimate infidelity $1-F$")
    fidelity_axis.set_title("E. Finite-shot accuracy improves")
    fidelity_axis.grid(which="both", alpha=0.25)
    fidelity_axis.legend(fontsize=7, ncol=2)

    physicality_axis = axes[1, 2]
    for method, marker in (("Linear", "o"), ("Projected", "s"), ("MLE", "^")):
        physicality_axis.semilogx(
            SHOT_COUNTS,
            analysis["nonphysical_fraction"][method],
            marker=marker,
            linewidth=2,
            color=method_colors[method],
            label=method,
        )
    physicality_axis.set_ylim(-0.04, 1.04)
    physicality_axis.set_xlabel("shots per Pauli setting")
    physicality_axis.set_ylabel("fraction with negative eigenvalue")
    physicality_axis.set_title("F. Physical estimators remain in state space")
    physicality_axis.grid(which="both", alpha=0.25)
    physicality_axis.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def save_results(
    path: Path,
    exact_records: list[dict[str, object]],
    finite_records: list[dict[str, object]],
    analysis: dict[str, object],
    representative: dict[str, object],
    trials: int,
    seed: int,
) -> None:
    """Save compact machine-readable evidence without density-matrix payloads."""
    summary_infidelity = {
        f"{family}|{method}": values.tolist()
        for (family, method), values in analysis["mean_infidelity"].items()
    }
    summary_physicality = {
        method: values.tolist()
        for method, values in analysis["nonphysical_fraction"].items()
    }
    payload = {
        "metadata": {
            "n_qubits": N_QUBITS,
            "shots_per_setting": list(SHOT_COUNTS),
            "trials_per_condition": trials,
            "seed": seed,
            "mle_max_iterations": MLE_MAX_ITERATIONS,
            "state_families": [family.name for family in STATE_FAMILIES],
        },
        "summary": {
            "maximum_exact_hs_distance": analysis["max_exact_error"],
            "representative_ghz_mle_fidelity": representative["fidelity"],
            "mean_infidelity": summary_infidelity,
            "nonphysical_fraction": summary_physicality,
        },
        "exact_records": exact_records,
        "finite_shot_records": finite_records,
        "representative_mle_history": np.asarray(
            representative["history"], dtype=float
        ).tolist(),
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
    exact_records = run_exact_closure(args.seed)
    finite_records, representative = run_finite_shot_trials(args.trials, args.seed)
    analysis = analyze(exact_records, finite_records)
    print_report(
        exact_records,
        finite_records,
        analysis,
        representative,
        args.trials,
    )
    plot_results(
        exact_records,
        analysis,
        representative,
        args.output,
        args.trials,
        show=args.show,
    )
    save_results(
        args.json,
        exact_records,
        finite_records,
        analysis,
        representative,
        args.trials,
        args.seed,
    )
    print(f"\nSaved figure: {args.output}")
    print(f"Saved data:   {args.json}")


if __name__ == "__main__":
    main()
