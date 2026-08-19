"""Merge synchronized benchmark CSVs and compare CPU/GPU scaling.

Example:
    python tools/compare_backend_timings.py \
      --inputs results/benchmark_numpy.csv results/benchmark_cupy.csv results/benchmark_jax.csv \
      --output results/backend_timing_summary.csv \
      --plot-dir results/backend_timing_plots
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


GROUP_FIELDS = (
    "backend",
    "device_name",
    "n_qubits",
    "state_type",
    "shots_per_setting",
    "method",
)
TIME_FIELDS = (
    "reconstruction_seconds",
    "fidelity_seconds",
    "method_total_seconds",
    "end_to_end_seconds",
)


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        raise ValueError("No benchmark rows were found")
    required = set(GROUP_FIELDS) | set(TIME_FIELDS) | {"fidelity", "is_physical", "fidelity_interpretable"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Input is not a synchronized timing CSV; missing {sorted(missing)}")
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        grouped[key].append(row)

    summary = []
    for key, selected in sorted(grouped.items()):
        output = dict(zip(GROUP_FIELDS, key))
        output["n_qubits"] = int(output["n_qubits"])
        output["shots_per_setting"] = int(output["shots_per_setting"])
        output["timed_samples"] = len(selected)
        output["mean_fidelity"] = float(np.mean([float(row["fidelity"]) for row in selected]))
        physical = [row["is_physical"].lower() == "true" for row in selected]
        interpretable = [
            float(row["fidelity"])
            for row in selected
            if row["fidelity_interpretable"].lower() == "true"
        ]
        output["physical_fraction"] = float(np.mean(physical))
        output["mean_interpretable_fidelity"] = (
            float(np.mean(interpretable)) if interpretable else "not_available"
        )
        for field in TIME_FIELDS:
            values = np.asarray([float(row[field]) for row in selected])
            stem = field.removesuffix("_seconds")
            output[f"mean_{stem}_seconds"] = float(np.mean(values))
            output[f"median_{stem}_seconds"] = float(np.median(values))
            output[f"std_{stem}_seconds"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            output[f"p95_{stem}_seconds"] = float(np.percentile(values, 95))
        summary.append(output)

    cpu_lookup = {}
    for row in summary:
        if row["backend"] == "numpy":
            key = (row["n_qubits"], row["state_type"], row["shots_per_setting"], row["method"])
            cpu_lookup[key] = row["median_method_total_seconds"]
    for row in summary:
        key = (row["n_qubits"], row["state_type"], row["shots_per_setting"], row["method"])
        cpu = cpu_lookup.get(key)
        current = row["median_method_total_seconds"]
        row["speedup_vs_numpy_median"] = float(cpu / current) if cpu is not None and current > 0 else "not_available"
    return summary


def write_csv(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows, plot_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to use --plot-dir") from error

    target = Path(plot_dir)
    target.mkdir(parents=True, exist_ok=True)
    state_types = sorted({row["state_type"] for row in rows})
    shot_values = sorted({int(row["shots_per_setting"]) for row in rows})
    for state_type in state_types:
        figure, axes = plt.subplots(
            2,
            len(shot_values),
            figsize=(5.0 * len(shot_values), 7.0),
            squeeze=False,
            constrained_layout=True,
        )
        for column, shots in enumerate(shot_values):
            selected = [
                row
                for row in rows
                if row["state_type"] == state_type and int(row["shots_per_setting"]) == shots
            ]
            labels = sorted({(row["backend"], row["method"]) for row in selected})
            for backend, method in labels:
                series = sorted(
                    (row for row in selected if row["backend"] == backend and row["method"] == method),
                    key=lambda row: int(row["n_qubits"]),
                )
                qubits = [int(row["n_qubits"]) for row in series]
                label = f"{backend} · {method.replace('_', ' ')}"
                axes[0, column].plot(
                    qubits,
                    [float(row["median_method_total_seconds"]) for row in series],
                    marker="o",
                    label=label,
                )
                interpretable_series = [
                    row for row in series if row["mean_interpretable_fidelity"] != "not_available"
                ]
                if interpretable_series:
                    axes[1, column].plot(
                        [int(row["n_qubits"]) for row in interpretable_series],
                        [float(row["mean_interpretable_fidelity"]) for row in interpretable_series],
                        marker="o",
                        label=label,
                    )
            axes[0, column].set_yscale("log")
            axes[0, column].set_title(f"{shots} shots per setting")
            axes[0, column].set_xlabel("qubits")
            axes[0, column].set_ylabel("median reconstruction + fidelity (s)")
            axes[1, column].set_xlabel("qubits")
            axes[1, column].set_ylabel("mean fidelity (physical estimates only)")
            axes[1, column].set_ylim(0.0, 1.02)
            for axis in axes[:, column]:
                axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.03),
                ncol=min(3, len(handles)),
            )
        figure.suptitle(f"Synchronized backend scaling · {state_type} states")
        figure.savefig(target / f"backend_scaling_{state_type}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--inputs", nargs="+", required=True)
    result.add_argument("--output", default="results/backend_timing_summary.csv")
    result.add_argument("--plot-dir")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    summary = summarize(read_rows(args.inputs))
    write_csv(args.output, summary)
    if args.plot_dir:
        make_plots(summary, args.plot_dir)
    print(f"Wrote {len(summary)} backend/method scaling cells to {args.output}")


if __name__ == "__main__":
    main()
