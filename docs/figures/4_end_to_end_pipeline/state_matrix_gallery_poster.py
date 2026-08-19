"""Create a poster-ready version of the four-qubit state-matrix gallery.

This keeps the original gallery data and colormaps, removes the overall title,
and enlarges panel labels, axis titles, ticks, metrics, and colorbar text.

Run from the repository root:

    python verification/4_end_to_end_pipeline/state_matrix_gallery_poster.py
"""

from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT = Path(__file__).with_name("state_matrix_gallery_poster.png")
TEXT_COLOR = "#0A2239"


def configure_poster_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.titlesize": 25,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_gallery_poster(
    rows: list[dict[str, object]],
    output: Path,
    *,
    show: bool = False,
) -> None:
    configure_poster_style()

    figure = plt.figure(figsize=(26.5, 20.5), layout="constrained")
    grid = figure.add_gridspec(
        len(rows),
        7,
        width_ratios=(0.13, 1.0, 1.0, 0.045, 1.0, 1.0, 0.045),
        hspace=0.05,
        wspace=0.08,
    )

    column_titles = (
        "Generated",
        "Measurement",
        "Linear inversion",
        "MLE",
    )
    matrix_grid_columns = (1, 4, 5)
    tick_positions = (0, 5, 10, 15)
    tick_labels = ("0000", "0101", "1010", "1111")

    for row_index, row in enumerate(rows):
        matrices = (
            np.asarray(row["target"]),
            np.asarray(row["measured"]),
            np.asarray(row["reconstructed"]),
        )
        row_vmax = max(float(np.max(np.abs(matrix))) for matrix in matrices)
        row_axes: list[plt.Axes] = []
        image = None

        for matrix_index, (matrix, grid_column) in enumerate(
            zip(matrices, matrix_grid_columns)
        ):
            axis = figure.add_subplot(grid[row_index, grid_column])
            row_axes.append(axis)
            image = axis.imshow(
                np.abs(matrix),
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=row_vmax,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_xticks(tick_positions, labels=tick_labels, rotation=30)
            axis.set_yticks(tick_positions, labels=tick_labels)
            axis.tick_params(axis="both", length=7, width=1.5, pad=7)
            if matrix_index != 0:
                axis.tick_params(axis="y", labelleft=False)

            if row_index == 0:
                title_index = matrix_index if matrix_index == 0 else matrix_index + 1
                axis.set_title(column_titles[title_index], pad=17)

        frequencies = np.asarray(row["frequencies"])
        frequency_axis = figure.add_subplot(grid[row_index, 2])
        frequency_image = frequency_axis.imshow(
            frequencies,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=float(np.max(frequencies)),
            interpolation="nearest",
            aspect="auto",
        )
        setting_ticks = np.linspace(0, frequencies.shape[0] - 1, 5, dtype=int)
        frequency_axis.set_yticks(setting_ticks)
        frequency_axis.set_yticklabels(
            [row["settings"][index] for index in setting_ticks]
        )
        frequency_axis.set_xticks(
            tick_positions,
            labels=tick_labels,
            rotation=30,
        )
        frequency_axis.tick_params(axis="both", length=7, width=1.5, pad=7)
        if row_index == 0:
            frequency_axis.set_title(column_titles[1], pad=17)

        frequency_colorbar_axis = figure.add_subplot(grid[row_index, 3])
        frequency_colorbar = figure.colorbar(
            frequency_image,
            cax=frequency_colorbar_axis,
        )
        frequency_colorbar.ax.tick_params(labelsize=18, width=1.3, length=6)

        row_label_axis = figure.add_subplot(grid[row_index, 0])
        row_label_axis.axis("off")
        row_label_axis.text(
            0.5,
            0.5,
            str(row["name"]),
            transform=row_label_axis.transAxes,
            ha="center",
            va="center",
            fontsize=23,
            fontweight="bold",
            rotation=90,
        )
        colorbar_axis = figure.add_subplot(grid[row_index, 6])
        matrix_colorbar = figure.colorbar(image, cax=colorbar_axis)
        matrix_colorbar.ax.tick_params(labelsize=18, width=1.3, length=6)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.10)
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mle-iterations",
        type=int,
        default=DEFAULT_MLE_ITERATIONS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_gallery_data(args.shots, args.seed, args.mle_iterations)
    print_report(rows, args.shots)
    plot_gallery_poster(rows, args.output, show=args.show)
    print(f"\nSaved poster figure: {args.output}")


if __name__ == "__main__":
    main()
