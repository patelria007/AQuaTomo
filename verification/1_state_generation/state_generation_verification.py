"""Verify and visualize the four-qubit state generators.

Run from the repository root with::

    python verification/state_generation/state_generation_verification.py

The script performs numerical assertions before writing the figure.  A plot is
useful as a structural sanity check, while the assertions are what establish
Hermiticity, unit trace, positive semidefiniteness, purity, and requested rank.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nbqst.states import (  # noqa: E402
    haar_random_pure,
    random_mixed_state,
    random_product_state,
)


N_QUBITS = 4
DIMENSION = 2**N_QUBITS
DEFAULT_OUTPUT = Path(__file__).with_name("four_qubit_state_verification.png")
TOLERANCE = 1e-10


def one_qubit_reduced_state(rho: np.ndarray, qubit: int) -> np.ndarray:
    """Return the reduced density matrix of one most-significant-first qubit."""
    if not 0 <= qubit < N_QUBITS:
        raise ValueError(f"qubit must be between 0 and {N_QUBITS - 1}")

    remaining = [index for index in range(N_QUBITS) if index != qubit]
    tensor = rho.reshape((2,) * (2 * N_QUBITS))
    permutation = (
        [qubit]
        + remaining
        + [N_QUBITS + qubit]
        + [N_QUBITS + index for index in remaining]
    )
    tensor = np.transpose(tensor, permutation)
    tensor = tensor.reshape(2, 2 ** (N_QUBITS - 1), 2, 2 ** (N_QUBITS - 1))
    return np.trace(tensor, axis1=1, axis2=3)


def state_diagnostics(rho: np.ndarray) -> dict[str, object]:
    """Calculate the physicality and structure diagnostics used in the plot."""
    rho = np.asarray(rho)
    eigenvalues = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real
    local_purities = np.asarray(
        [
            np.trace(reduced @ reduced).real
            for reduced in (
                one_qubit_reduced_state(rho, qubit)
                for qubit in range(N_QUBITS)
            )
        ]
    )
    return {
        "trace": np.trace(rho),
        "hermitian_error": np.linalg.norm(rho - rho.conj().T),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "purity": float(np.trace(rho @ rho).real),
        "rank": int(np.count_nonzero(eigenvalues > TOLERANCE)),
        "eigenvalues": eigenvalues,
        "local_purities": local_purities,
    }


def assert_density_matrix(name: str, rho: np.ndarray) -> dict[str, object]:
    """Assert the defining density-matrix conditions and return diagnostics."""
    rho = np.asarray(rho)
    if rho.shape != (DIMENSION, DIMENSION):
        raise AssertionError(f"{name}: expected {(DIMENSION, DIMENSION)}, got {rho.shape}")

    diagnostics = state_diagnostics(rho)
    if diagnostics["hermitian_error"] > TOLERANCE:
        raise AssertionError(f"{name}: density matrix is not Hermitian")
    if abs(diagnostics["trace"] - 1.0) > TOLERANCE:
        raise AssertionError(f"{name}: trace is not one")
    if diagnostics["minimum_eigenvalue"] < -TOLERANCE:
        raise AssertionError(f"{name}: density matrix is not positive semidefinite")
    return diagnostics


def generate_and_verify_states() -> tuple[dict[str, object], ...]:
    """Generate reproducible four-qubit examples and verify their structure."""
    specifications = (
        (
            "Product pure",
            random_product_state(N_QUBITS, rng=101),
            1,
        ),
        (
            "Global Haar pure",
            haar_random_pure(N_QUBITS, rng=102),
            1,
        ),
        (
            "Mixed, rank=4",
            random_mixed_state(N_QUBITS, rank=4, rng=103),
            4,
        ),
        (
            "Mixed, rank=16",
            random_mixed_state(N_QUBITS, rank=16, rng=104),
            16,
        ),
    )

    verified = []
    for name, rho, expected_rank in specifications:
        rho = np.asarray(rho)
        diagnostics = assert_density_matrix(name, rho)
        if diagnostics["rank"] != expected_rank:
            raise AssertionError(
                f"{name}: expected rank {expected_rank}, got {diagnostics['rank']}"
            )
        if expected_rank == 1 and abs(diagnostics["purity"] - 1.0) > TOLERANCE:
            raise AssertionError(f"{name}: a pure state must have purity one")
        verified.append({"name": name, "rho": rho, **diagnostics})

    product_local_purities = verified[0]["local_purities"]
    if not np.allclose(product_local_purities, 1.0, atol=TOLERANCE):
        raise AssertionError("Product pure: every one-qubit reduced state must be pure")

    # With this fixed seed, the Haar state is entangled across every 1|3 split.
    # This is a structural sanity check, not a proof of the Haar distribution.
    if np.any(verified[1]["local_purities"] >= 1.0 - 1e-6):
        raise AssertionError("Global Haar pure: expected entanglement in this sample")

    return tuple(verified)


def print_report(states: tuple[dict[str, object], ...]) -> None:
    """Print the numerical values behind the visual checks."""
    header = (
        f"{'state':<20} {'trace error':>12} {'Hermitian':>12} "
        f"{'min eig':>12} {'purity':>10} {'rank':>6}"
    )
    print(header)
    print("-" * len(header))
    for state in states:
        trace_error = abs(state["trace"] - 1.0)
        print(
            f"{state['name']:<20} {trace_error:12.3e} "
            f"{state['hermitian_error']:12.3e} "
            f"{state['minimum_eigenvalue']:12.3e} "
            f"{state['purity']:10.6f} {state['rank']:6d}"
        )
    print("\nAll density-matrix and state-family assertions passed.")


def plot_states(
    states: tuple[dict[str, object], ...], output: Path, *, show: bool = False
) -> None:
    """Plot density-matrix magnitude, spectrum, and one-qubit purities."""
    figure, axes = plt.subplots(
        3,
        len(states),
        figsize=(15.5, 10.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.1, 0.85, 0.75)},
    )

    maximum_magnitude = max(np.abs(state["rho"]).max() for state in states)
    image = None
    basis_ticks = [0, 4, 8, 12, 15]
    basis_labels = [format(index, "04b") for index in basis_ticks]

    for column, state in enumerate(states):
        rho = state["rho"]
        image = axes[0, column].imshow(
            np.abs(rho),
            cmap="magma",
            vmin=0.0,
            vmax=maximum_magnitude,
            interpolation="nearest",
        )
        axes[0, column].set_title(state["name"])
        axes[0, column].set_xticks(basis_ticks, basis_labels, rotation=45)
        axes[0, column].set_yticks(basis_ticks, basis_labels)
        axes[0, column].set_xlabel(r"column basis state $|j\rangle$")
        if column == 0:
            axes[0, column].set_ylabel(r"row basis state $|i\rangle$")

        spectrum = np.sort(np.clip(state["eigenvalues"], 0.0, None))[::-1]
        axes[1, column].bar(np.arange(DIMENSION), spectrum, color="#3569a8")
        axes[1, column].set_xlim(-0.7, DIMENSION - 0.3)
        axes[1, column].set_ylim(0.0, 1.03)
        axes[1, column].set_xticks([0, 3, 7, 11, 15])
        axes[1, column].set_xlabel("sorted eigenvalue index")
        axes[1, column].set_title(
            f"rank={state['rank']}, purity={state['purity']:.4f}"
        )
        if column == 0:
            axes[1, column].set_ylabel("eigenvalue")
        axes[1, column].grid(axis="y", alpha=0.25)

        local_purities = state["local_purities"]
        axes[2, column].bar(
            np.arange(N_QUBITS),
            local_purities,
            color="#cf6a32",
        )
        axes[2, column].axhline(0.5, color="0.35", linestyle="--", linewidth=1)
        axes[2, column].set_ylim(0.45, 1.03)
        axes[2, column].set_xticks(range(N_QUBITS), [f"q{q}" for q in range(N_QUBITS)])
        axes[2, column].set_xlabel("one-qubit subsystem")
        if column == 0:
            axes[2, column].set_ylabel(r"local purity $\mathrm{Tr}(\rho_q^2)$")
        axes[2, column].grid(axis="y", alpha=0.25)

    colorbar = figure.colorbar(image, ax=axes[0, :], shrink=0.75, pad=0.02)
    colorbar.set_label(r"density-matrix magnitude $|\rho_{ij}|$ (shared scale)")
    figure.suptitle("Four-qubit state-generation verification", fontsize=16)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"figure path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the saved figure after verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states = generate_and_verify_states()
    print_report(states)
    plot_states(states, args.output, show=args.show)
    print(f"Figure written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
