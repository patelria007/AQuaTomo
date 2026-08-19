"""Plot 3-qubit reconstruction fidelity versus finite-shot count.

The experiment compares random product-pure, Haar-random pure, and full-rank
Ginibre mixed states.  Every data point is an ensemble mean over independently
seeded target states and multinomial measurement records.  Measurements use all
3**3 local-Pauli settings.  Symmetric readout fidelity can optionally be applied
without mitigation.  Reconstruction is linear inversion followed by the
nearest positive-semidefinite, unit-trace projection, so the same estimator is
used for all three state families.

Run from the repository root with::

    python "verification/shot vs fidelity/shot_vs_fidelity.py"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nbqst.denoise import project_density_matrix  # noqa: E402
from nbqst.measurements import complete_pauli_settings, simulate_pauli_measurements  # noqa: E402
from nbqst.metrics import fidelity, minimum_eigenvalue  # noqa: E402
from nbqst.reconstruction import linear_inversion_pauli  # noqa: E402
from nbqst.states import (  # noqa: E402
    haar_random_pure,
    random_mixed_state,
    random_product_state,
)


N_QUBITS = 3
N_SETTINGS = 3**N_QUBITS
SHOT_COUNTS = (100, 150, 220, 330, 470, 680, 1_000, 1_500, 2_200, 3_300, 4_700, 6_800, 10_000)
TARGET_FIDELITY = 0.99
DEFAULT_TRIALS = 100
DEFAULT_SEED = 20260818
OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = OUTPUT_DIR / "shot_vs_fidelity.png"
DEFAULT_CSV = OUTPUT_DIR / "shot_vs_fidelity_results.csv"
DEFAULT_JSON = OUTPUT_DIR / "shot_vs_fidelity_results.json"

STATE_FAMILIES = (
    ("Product", random_product_state),
    ("Pure (Haar)", haar_random_pure),
    ("Mixed (full rank)", random_mixed_state),
)
STATE_FAMILY_MAP = {
    "product": ("Product", random_product_state),
    "pure": ("Pure (Haar)", haar_random_pure),
    "mixed": ("Mixed (full rank)", random_mixed_state),
}


def seeded_rng(seed: int, *keys: int) -> np.random.Generator:
    """Return an RNG stream whose identity does not depend on loop order."""

    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def collect_trials(
    trials: int,
    seed: int,
    readout_fidelity_0: float | None = None,
    readout_fidelity_1: float | None = None,
) -> list[dict[str, object]]:
    """Simulate and reconstruct every family/shot/trial combination."""

    if trials < 2:
        raise ValueError("trials must be at least 2 to estimate uncertainty")

    expected_settings = complete_pauli_settings(N_QUBITS)
    records: list[dict[str, object]] = []
    for family_index, (family_name, generator) in enumerate(STATE_FAMILIES):
        for trial in range(trials):
            # Reuse the same target across shot counts within a trial.  Only the
            # multinomial record changes, which makes shot scaling easier to read.
            state_rng = seeded_rng(seed, 11, family_index, trial)
            rho_true = np.asarray(generator(N_QUBITS, rng=state_rng))

            for shots_per_setting in SHOT_COUNTS:
                measurement_rng = seeded_rng(
                    seed, 29, family_index, trial, shots_per_setting
                )
                data = simulate_pauli_measurements(
                    rho_true,
                    shots_per_setting,
                    rng=measurement_rng,
                    readout_fidelity_0=readout_fidelity_0,
                    readout_fidelity_1=readout_fidelity_1,
                )
                if data.settings != expected_settings:
                    raise AssertionError("the complete 27-setting Pauli design changed")
                if any(
                    int(np.sum(counts)) != shots_per_setting
                    for counts in data.counts.values()
                ):
                    raise AssertionError("a setting did not conserve its shot count")

                linear = np.asarray(linear_inversion_pauli(data))
                estimate = np.asarray(project_density_matrix(linear))
                if float(minimum_eigenvalue(estimate)) < -1e-10:
                    raise AssertionError("the physical projection returned a non-PSD state")

                records.append(
                    {
                        "family": family_name,
                        "trial": trial,
                        "shots_per_setting": shots_per_setting,
                        "total_shots": shots_per_setting * N_SETTINGS,
                        "fidelity": float(fidelity(rho_true, estimate)),
                    }
                )
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return mean fidelity and a normal 95% confidence interval per point."""

    summary: list[dict[str, object]] = []
    for family_name, _ in STATE_FAMILIES:
        for shots_per_setting in SHOT_COUNTS:
            values = np.asarray(
                [
                    row["fidelity"]
                    for row in records
                    if row["family"] == family_name
                    and row["shots_per_setting"] == shots_per_setting
                ],
                dtype=float,
            )
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1))
            standard_error = standard_deviation / np.sqrt(values.size)
            half_width = 1.96 * standard_error
            summary.append(
                {
                    "family": family_name,
                    "shots_per_setting": shots_per_setting,
                    "total_shots": shots_per_setting * N_SETTINGS,
                    "trials": int(values.size),
                    "mean_fidelity": mean,
                    "standard_deviation": standard_deviation,
                    "standard_error": standard_error,
                    "ci95_lower": max(0.0, mean - half_width),
                    "ci95_upper": min(1.0, mean + half_width),
                }
            )
    return summary


def threshold_summary(summary: list[dict[str, object]]) -> dict[str, dict[str, int | None]]:
    """Find the first tested shot count whose ensemble mean reaches 99%."""

    thresholds: dict[str, dict[str, int | None]] = {}
    for family_name, _ in STATE_FAMILIES:
        family_rows = [row for row in summary if row["family"] == family_name]
        crossing = next(
            (row for row in family_rows if row["mean_fidelity"] >= TARGET_FIDELITY),
            None,
        )
        thresholds[family_name] = {
            "shots_per_setting": (
                int(crossing["shots_per_setting"]) if crossing is not None else None
            ),
            "total_shots": int(crossing["total_shots"]) if crossing is not None else None,
        }
    return thresholds


def plot_results(
    summary: list[dict[str, object]],
    thresholds: dict[str, dict[str, int | None]],
    output: Path,
    trials: int,
    readout_fidelity_0: float | None,
    readout_fidelity_1: float | None,
    *,
    show: bool = False,
) -> None:
    """Draw the three fidelity curves and the 99% target on one axis."""

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
    styles = {
        "Product": {"color": "#2878B5", "marker": "o"},
        "Pure (Haar)": {"color": "#9B59B6", "marker": "s"},
        "Mixed (full rank)": {"color": "#D35400", "marker": "^"},
    }

    all_lower_bounds = []
    for family_name, _ in STATE_FAMILIES:
        rows = [row for row in summary if row["family"] == family_name]
        shots = np.asarray([row["shots_per_setting"] for row in rows], dtype=float)
        means = np.asarray([row["mean_fidelity"] for row in rows], dtype=float)
        lowers = np.asarray([row["ci95_lower"] for row in rows], dtype=float)
        uppers = np.asarray([row["ci95_upper"] for row in rows], dtype=float)
        all_lower_bounds.extend(lowers)
        style = styles[family_name]
        axis.plot(
            shots,
            means,
            linewidth=2.2,
            markersize=5.5,
            label=family_name,
            **style,
        )
        axis.fill_between(shots, lowers, uppers, color=style["color"], alpha=0.13)

        crossing_shots = thresholds[family_name]["shots_per_setting"]
        if crossing_shots is not None:
            crossing_row = next(
                row for row in rows if row["shots_per_setting"] == crossing_shots
            )
            axis.scatter(
                [crossing_shots],
                [crossing_row["mean_fidelity"]],
                s=82,
                facecolors="none",
                edgecolors=style["color"],
                linewidths=2.0,
                zorder=5,
            )

    axis.axhline(
        TARGET_FIDELITY,
        color="0.22",
        linewidth=1.5,
        linestyle="--",
        label="99% fidelity target",
    )
    axis.set_xscale("log")
    axis.set_xlim(SHOT_COUNTS[0], SHOT_COUNTS[-1])
    axis.set_xticks((100, 200, 500, 1_000, 2_000, 5_000, 10_000))
    axis.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{int(value):,}")
    )
    lower_limit = max(0.0, min(all_lower_bounds) - 0.015)
    axis.set_ylim(lower_limit, 1.003)
    axis.set_xlabel(
        f"Shots per Pauli setting ({N_SETTINGS} settings; "
        f"total shots = {N_SETTINGS} × x)"
    )
    axis.set_ylabel("Mean reconstruction fidelity")
    if readout_fidelity_0 is None:
        readout_title = "Without Readout Error"
    elif np.isclose(readout_fidelity_0, readout_fidelity_1):
        readout_title = (
            f"With {100.0 * readout_fidelity_0:.1f}% Symmetric Readout Fidelity"
        )
    else:
        readout_title = (
            "With Asymmetric Readout Fidelity "
            f"(F0={100.0 * readout_fidelity_0:.1f}%, "
            f"F1={100.0 * readout_fidelity_1:.1f}%)"
        )
    axis.set_title(
        f"{N_QUBITS}-Qubit Tomography {readout_title}\n"
        "Physical linear inversion (mean ± 95% CI)"
    )
    axis.grid(which="major", alpha=0.28)
    axis.grid(which="minor", axis="x", alpha=0.12)
    axis.legend(loc="lower right", frameon=True)

    threshold_lines = []
    for family_name, _ in STATE_FAMILIES:
        shot_value = thresholds[family_name]["shots_per_setting"]
        text = f"{shot_value:,}/setting" if shot_value is not None else ">10,000/setting"
        threshold_lines.append(f"{family_name}: {text}")
    axis.text(
        0.025,
        0.035,
        "First tested mean ≥ 99%\n" + "\n".join(threshold_lines),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
    )
    figure.text(
        0.995,
        0.005,
        f"{trials} random target states per family; squared Uhlmann fidelity",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.35",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_csv(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def save_json(
    path: Path,
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
    thresholds: dict[str, dict[str, int | None]],
    trials: int,
    seed: int,
    readout_fidelity_0: float | None,
    readout_fidelity_1: float | None,
) -> None:
    payload = {
        "metadata": {
            "n_qubits": N_QUBITS,
            "settings": N_SETTINGS,
            "shot_definition": "shots per local-Pauli setting",
            "shot_counts": list(SHOT_COUNTS),
            "trials_per_family": trials,
            "seed": seed,
            "readout_fidelity_0": readout_fidelity_0,
            "readout_fidelity_1": readout_fidelity_1,
            "readout_error_mitigation": None,
            "estimator": "Pauli linear inversion + PSD unit-trace projection",
            "fidelity": "squared Uhlmann fidelity",
            "target_fidelity": TARGET_FIDELITY,
            "threshold_rule": "first tested shot count with ensemble mean fidelity >= target",
        },
        "thresholds": thresholds,
        "summary": summary,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_report(
    summary: list[dict[str, object]],
    thresholds: dict[str, dict[str, int | None]],
    readout_fidelity_0: float | None,
    readout_fidelity_1: float | None,
) -> None:
    if readout_fidelity_0 is None:
        readout_text = "no readout error"
    elif np.isclose(readout_fidelity_0, readout_fidelity_1):
        readout_text = (
            f"symmetric readout fidelity={readout_fidelity_0:.4f}, no mitigation"
        )
    else:
        readout_text = (
            f"asymmetric readout F0={readout_fidelity_0:.4f}, "
            f"F1={readout_fidelity_1:.4f}, no mitigation"
        )
    print(f"{N_QUBITS}-qubit shot-versus-fidelity experiment ({readout_text})")
    print(f"Estimator: projected linear inversion; settings: {N_SETTINGS}")
    print("\nFirst tested ensemble mean at or above 99% fidelity")
    for family_name, _ in STATE_FAMILIES:
        threshold = thresholds[family_name]
        if threshold["shots_per_setting"] is None:
            print(f"  {family_name:<18} > {SHOT_COUNTS[-1]:,} shots/setting")
        else:
            print(
                f"  {family_name:<18} {threshold['shots_per_setting']:>6,} shots/setting "
                f"({threshold['total_shots']:>7,} total shots)"
            )

    print("\nMean fidelity at 10,000 shots/setting")
    for family_name, _ in STATE_FAMILIES:
        row = next(
            row
            for row in summary
            if row["family"] == family_name
            and row["shots_per_setting"] == SHOT_COUNTS[-1]
        )
        print(
            f"  {family_name:<18} {row['mean_fidelity']:.6f} "
            f"(95% CI {row['ci95_lower']:.6f}-{row['ci95_upper']:.6f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qubits", type=int, default=N_QUBITS)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(STATE_FAMILY_MAP),
        default=tuple(STATE_FAMILY_MAP),
        help="state families to include (default: product pure mixed)",
    )
    parser.add_argument(
        "--shot-counts",
        nargs="+",
        type=int,
        default=None,
        help="custom shots-per-setting grid",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--readout-fidelity",
        type=float,
        default=None,
        help="symmetric P(0|0)=P(1|1); omit for ideal readout",
    )
    parser.add_argument(
        "--readout-fidelity-0",
        type=float,
        default=None,
        help="P(measured 0 | true 0); use with --readout-fidelity-1",
    )
    parser.add_argument(
        "--readout-fidelity-1",
        type=float,
        default=None,
        help="P(measured 1 | true 1); use with --readout-fidelity-0",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    global N_QUBITS, N_SETTINGS, SHOT_COUNTS, STATE_FAMILIES

    args = parse_args()
    if args.qubits < 1:
        raise ValueError("--qubits must be positive")
    N_QUBITS = args.qubits
    N_SETTINGS = 3**N_QUBITS
    STATE_FAMILIES = tuple(STATE_FAMILY_MAP[key] for key in args.families)
    if args.shot_counts is not None:
        if any(value < 1 for value in args.shot_counts):
            raise ValueError("--shot-counts values must be positive")
        SHOT_COUNTS = tuple(sorted(set(args.shot_counts)))
        if len(SHOT_COUNTS) < 2:
            raise ValueError("--shot-counts must contain at least two distinct values")
    separate_supplied = (
        args.readout_fidelity_0 is not None or args.readout_fidelity_1 is not None
    )
    if args.readout_fidelity is not None and separate_supplied:
        raise ValueError(
            "use either --readout-fidelity or the separate fidelity-0/fidelity-1 flags"
        )
    if (args.readout_fidelity_0 is None) != (args.readout_fidelity_1 is None):
        raise ValueError(
            "--readout-fidelity-0 and --readout-fidelity-1 must be provided together"
        )
    if args.readout_fidelity is not None:
        readout_fidelity_0 = readout_fidelity_1 = args.readout_fidelity
    else:
        readout_fidelity_0 = args.readout_fidelity_0
        readout_fidelity_1 = args.readout_fidelity_1
    for value in (readout_fidelity_0, readout_fidelity_1):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("readout fidelities must lie between 0 and 1")

    records = collect_trials(
        args.trials,
        args.seed,
        readout_fidelity_0,
        readout_fidelity_1,
    )
    summary = summarize(records)
    thresholds = threshold_summary(summary)
    print_report(summary, thresholds, readout_fidelity_0, readout_fidelity_1)
    plot_results(
        summary,
        thresholds,
        args.output,
        args.trials,
        readout_fidelity_0,
        readout_fidelity_1,
        show=args.show,
    )
    save_csv(args.csv, summary)
    save_json(
        args.json,
        records,
        summary,
        thresholds,
        args.trials,
        args.seed,
        readout_fidelity_0,
        readout_fidelity_1,
    )
    print(f"\nSaved figure: {args.output.resolve()}")
    print(f"Saved CSV:    {args.csv.resolve()}")
    print(f"Saved JSON:   {args.json.resolve()}")


if __name__ == "__main__":
    main()
