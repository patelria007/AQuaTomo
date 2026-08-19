"""Export the state-matrix gallery as 16 poster-ready PNG panels.

The four states and four tomography stages are saved separately so they can be
arranged freely in PowerPoint.  Panel titles and metric captions are omitted;
filenames identify each panel.

Run from the repository root:

    python verification/4_end_to_end_pipeline/state_matrix_gallery_panels.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from state_matrix_gallery import (
    DEFAULT_MLE_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_SHOTS,
    build_gallery_data,
    print_report,
)


DEFAULT_OUTPUT_DIR = Path(__file__).with_name("state_matrix_gallery_panels")
TEXT_COLOR = "#0A2239"
TICK_POSITIONS = (0, 5, 10, 15)
TICK_LABELS = ("0000", "0101", "1010", "1111")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 21,
            "axes.labelsize": 25,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def style_axis(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", length=7, width=1.5, pad=7)
    for spine in axis.spines.values():
        spine.set_linewidth(1.4)


def save_matrix_panel(
    matrix: np.ndarray,
    vmax: float,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 6.4), dpi=300)
    image = axis.imshow(
        np.abs(matrix),
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    axis.set_xticks(TICK_POSITIONS, labels=TICK_LABELS, rotation=30)
    axis.set_yticks(TICK_POSITIONS, labels=TICK_LABELS)
    style_axis(axis)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.055, pad=0.055)
    colorbar.ax.tick_params(labelsize=20, width=1.3, length=6)
    colorbar.outline.set_linewidth(1.2)

    figure.subplots_adjust(left=0.16, right=0.88, bottom=0.11, top=0.98)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def save_pauli_panel(
    frequencies: np.ndarray,
    settings: tuple[str, ...] | list[str],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 6.4), dpi=300)
    image = axis.imshow(
        frequencies,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.max(frequencies)),
        interpolation="nearest",
        aspect="auto",
    )

    setting_ticks = np.linspace(0, frequencies.shape[0] - 1, 5, dtype=int)
    axis.set_yticks(setting_ticks)
    axis.set_yticklabels([settings[index] for index in setting_ticks])
    axis.set_xticks(TICK_POSITIONS, labels=TICK_LABELS, rotation=30)
    style_axis(axis)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.055, pad=0.055)
    colorbar.ax.tick_params(labelsize=20, width=1.3, length=6)
    colorbar.outline.set_linewidth(1.2)

    figure.subplots_adjust(left=0.16, right=0.87, bottom=0.12, top=0.98)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def export_panels(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for state_index, row in enumerate(rows, start=1):
        state_slug = slugify(str(row["name"]))
        matrices = (
            np.asarray(row["target"]),
            np.asarray(row["measured"]),
            np.asarray(row["reconstructed"]),
        )
        row_vmax = max(float(np.max(np.abs(matrix))) for matrix in matrices)

        panel_specs = (
            ("01_generated", matrices[0]),
            ("03_linear_inversion", matrices[1]),
            ("04_reconstruction", matrices[2]),
        )
        for panel_name, matrix in panel_specs:
            output = output_dir / (
                f"{state_index:02d}_{state_slug}__{panel_name}.png"
            )
            save_matrix_panel(matrix, row_vmax, output)
            outputs.append(output)

        pauli_output = output_dir / (
            f"{state_index:02d}_{state_slug}__02_pauli_data.png"
        )
        save_pauli_panel(
            np.asarray(row["frequencies"]),
            row["settings"],
            pauli_output,
        )
        outputs.append(pauli_output)

    return sorted(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mle-iterations",
        type=int,
        default=DEFAULT_MLE_ITERATIONS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_gallery_data(args.shots, args.seed, args.mle_iterations)
    print_report(rows, args.shots)
    outputs = export_panels(rows, args.output_dir)
    print(f"\nSaved {len(outputs)} poster panels to: {args.output_dir}")
    for output in outputs:
        print(f"  {output.name}")


if __name__ == "__main__":
    main()
