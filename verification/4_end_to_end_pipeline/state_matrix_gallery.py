"""Plot four-qubit targets, complete Pauli data, and reconstructed matrices.

Run from the repository root with::

    python verification/4_end_to_end_pipeline/state_matrix_gallery.py

The middle column is the raw Pauli linear-inversion matrix obtained directly
from finite-shot counts.  It is therefore a density-matrix representation of
the measured data, not an independently generated quantum state.
"""

from __future__ import annotations

import argparse
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
from nbqst.measurements import simulate_pauli_measurements  # noqa: E402
from nbqst.metrics import fidelity, hilbert_schmidt_distance, purity  # noqa: E402
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli  # noqa: E402
from nbqst.states import (  # noqa: E402
    ghz_state,
    haar_random_pure,
    random_mixed_state,
    random_product_state,
)


N_QUBITS = 4
DEFAULT_SHOTS = 4096
DEFAULT_SEED = 20260921
DEFAULT_MLE_ITERATIONS = 100
DEFAULT_OUTPUT = Path(__file__).with_name("state_matrix_gallery.png")
PHYSICAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class StateExample:
    name: str
    build: Callable[[int], np.ndarray]


def state_seed(seed: int, state_index: int) -> int:
    sequence = np.random.SeedSequence([seed, 11, state_index, N_QUBITS])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def measurement_rng(seed: int, state_index: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([seed, 29, state_index, N_QUBITS])
    )


def state_examples(seed: int) -> tuple[StateExample, ...]:
    return (
        StateExample(
            "Product pure",
            lambda index: np.asarray(
                random_product_state(N_QUBITS, rng=state_seed(seed, index))
            ),
        ),
        StateExample(
            "Haar pure",
            lambda index: np.asarray(
                haar_random_pure(N_QUBITS, rng=state_seed(seed, index))
            ),
        ),
        StateExample(
            "Rank-4 mixed",
            lambda index: np.asarray(
                random_mixed_state(
                    N_QUBITS,
                    rank=4,
                    rng=state_seed(seed, index),
                )
            ),
        ),
        StateExample(
            "GHZ",
            lambda index: np.asarray(ghz_state(N_QUBITS)),
        ),
    )


def minimum_eigenvalue(matrix: np.ndarray) -> float:
    hermitian = (matrix + matrix.conj().T) / 2.0
    return float(np.linalg.eigvalsh(hermitian)[0])


def build_gallery_data(
    shots: int,
    seed: int,
    mle_iterations: int,
) -> list[dict[str, object]]:
    if shots < 1:
        raise ValueError("shots must be positive")
    if mle_iterations < 1:
        raise ValueError("mle_iterations must be positive")

    rows: list[dict[str, object]] = []
    for state_index, example in enumerate(state_examples(seed)):
        target = example.build(state_index)
        measurements = simulate_pauli_measurements(
            target,
            shots,
            rng=measurement_rng(seed, state_index),
        )
        measured = np.asarray(linear_inversion_pauli(measurements))
        initial = np.asarray(project_density_matrix(measured))
        reconstructed = np.asarray(
            factorized_mle(
                measurements,
                initial=initial,
                max_iter=mle_iterations,
                learning_rate=0.25,
            )
        )

        reconstructed_minimum = minimum_eigenvalue(reconstructed)
        if reconstructed_minimum < -PHYSICAL_TOLERANCE:
            raise AssertionError(
                f"{example.name} MLE reconstruction is not physical: "
                f"minimum eigenvalue={reconstructed_minimum:.3e}"
            )

        rows.append(
            {
                "name": example.name,
                "target": target,
                "frequencies": np.stack(
                    [
                        np.asarray(measurements.counts[setting], dtype=float)
                        / shots
                        for setting in measurements.settings
                    ]
                ),
                "settings": measurements.settings,
                "measured": measured,
                "reconstructed": reconstructed,
                "target_purity": float(purity(target)),
                "measured_hs_distance": float(
                    hilbert_schmidt_distance(target, measured)
                ),
                "measured_minimum_eigenvalue": minimum_eigenvalue(measured),
                "reconstructed_fidelity": float(fidelity(target, reconstructed)),
                "reconstructed_minimum_eigenvalue": reconstructed_minimum,
            }
        )
    return rows


def plot_gallery(
    rows: list[dict[str, object]],
    output: Path,
    shots: int,
    *,
    show: bool = False,
) -> None:
    figure = plt.figure(figsize=(19.5, 16.5), layout="constrained")
    grid = figure.add_gridspec(
        len(rows),
        6,
        width_ratios=(1.0, 1.0, 0.04, 1.0, 1.0, 0.04),
    )
    figure.suptitle(
        (
            "Four-qubit tomography: generated, measured, and reconstructed "
            f"matrices ({shots:,} shots/setting)"
        ),
        fontsize=20,
    )

    column_titles = (
        "Generated target",
        "Complete Pauli data",
        "Measured (raw linear inversion)",
        "Reconstructed (physical MLE)",
    )
    matrix_grid_columns = (0, 3, 4)
    tick_positions = (0, 5, 10, 15)
    tick_labels = ("0000", "0101", "1010", "1111")

    for row_index, row in enumerate(rows):
        matrices = (
            np.asarray(row["target"]),
            np.asarray(row["measured"]),
            np.asarray(row["reconstructed"]),
        )
        row_vmax = max(float(np.max(np.abs(matrix))) for matrix in matrices)
        row_axes = []
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
            axis.set_xticks(tick_positions, labels=tick_labels, rotation=35)
            axis.set_yticks(tick_positions, labels=tick_labels)
            axis.set_xlabel("computational-basis column")
            if matrix_index == 0:
                axis.set_ylabel("computational-basis row")
            else:
                axis.tick_params(axis="y", labelleft=False)
            if row_index == 0:
                axis.set_title(
                    column_titles[matrix_index if matrix_index == 0 else matrix_index + 1],
                    fontsize=13,
                    pad=10,
                )

        frequencies = np.asarray(row["frequencies"])
        frequency_axis = figure.add_subplot(grid[row_index, 1])
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
        frequency_axis.set_xticks(tick_positions, labels=tick_labels, rotation=35)
        frequency_axis.set_xlabel("measurement outcome")
        frequency_axis.set_ylabel("Pauli setting (81 total)")
        if row_index == 0:
            frequency_axis.set_title(column_titles[1], fontsize=13, pad=10)

        frequency_colorbar_axis = figure.add_subplot(grid[row_index, 2])
        figure.colorbar(
            frequency_image,
            cax=frequency_colorbar_axis,
            label="observed frequency",
        )

        row_axes[0].text(
            -0.23,
            0.5,
            str(row["name"]),
            transform=row_axes[0].transAxes,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="semibold",
            rotation=90,
        )
        row_axes[0].text(
            0.5,
            -0.24,
            f"purity = {row['target_purity']:.3f}",
            transform=row_axes[0].transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        row_axes[1].text(
            0.5,
            -0.24,
            (
                f"HS error = {row['measured_hs_distance']:.3f}; "
                f"$\\lambda_{{\\min}}$ = "
                f"{row['measured_minimum_eigenvalue']:.2e}"
            ),
            transform=row_axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        row_axes[2].text(
            0.5,
            -0.24,
            (
                f"fidelity = {row['reconstructed_fidelity']:.5f}; "
                f"$\\lambda_{{\\min}}$ = "
                f"{row['reconstructed_minimum_eigenvalue']:.2e}"
            ),
            transform=row_axes[2].transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

        colorbar_axis = figure.add_subplot(grid[row_index, 5])
        figure.colorbar(image, cax=colorbar_axis, label=r"$|\rho_{ij}|$")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def print_report(rows: list[dict[str, object]], shots: int) -> None:
    print(f"Four-qubit state-matrix gallery ({shots:,} shots/setting)")
    print("State                 raw HS error   raw min eig   MLE fidelity")
    for row in rows:
        print(
            f"{row['name']:<21} "
            f"{row['measured_hs_distance']:>12.5f}   "
            f"{row['measured_minimum_eigenvalue']:>11.3e}   "
            f"{row['reconstructed_fidelity']:>12.6f}"
        )


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
    plot_gallery(rows, args.output, args.shots, show=args.show)
    print(f"\nSaved figure: {args.output}")


if __name__ == "__main__":
    main()
