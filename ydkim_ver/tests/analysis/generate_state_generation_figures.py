"""Generate reproducible poster figures for the state-generation milestone.

This analysis script may use NumPy and Matplotlib; the core generator remains
Array-API portable.  All random states are obtained through the public API,
whose sole random source is stdlib ``random.Random``.

AI disclosure: this plotting code and its visual design were generated with
OpenAI Codex assistance on 2026-08-17.  Outputs require independent review.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nbqs_qst.state_generation import (  # noqa: E402
    random_haar_state,
    random_mixed_state,
    random_product_state,
    random_state_with_purity,
)


OUTPUT_DIR = ROOT / "docs" / "figures" / "state_generation"
LIKE = np.asarray(0.0)
BLUE = "#176B87"
ORANGE = "#E07A3F"
GREEN = "#3A8D72"
PURPLE = "#7656A5"
DARK = "#25313C"
GRID = "#CBD3D9"


def _configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _bloch_vector(ket):
    a, b = np.asarray(ket)
    coherence = np.conj(a) * b
    return np.asarray(
        [2 * coherence.real, 2 * coherence.imag, abs(a) ** 2 - abs(b) ** 2]
    )


def _bipartite_entropy_bits(ket):
    vector = np.asarray(ket)
    subsystem_dimension = int(round(math.sqrt(vector.size)))
    coefficient_matrix = vector.reshape(subsystem_dimension, subsystem_dimension)
    probabilities = np.linalg.svd(coefficient_matrix, compute_uv=False) ** 2
    probabilities = probabilities[probabilities > 1e-15]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _page_entropy_bits(m, k):
    entropy_nats = sum(1 / index for index in range(k + 1, m * k + 1))
    entropy_nats -= (m - 1) / (2 * k)
    return entropy_nats / math.log(2)


def _save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_families_one_to_six_qubits():
    family_specs = (
        ("Product\npure", lambda n, seed: random_product_state(LIKE, n, seed=seed)),
        ("Haar\npure", lambda n, seed: random_haar_state(LIKE, n, seed=seed)),
        (
            "Induced\nmixed",
            lambda n, seed: random_mixed_state(LIKE, n, k=2**n, seed=seed),
        ),
    )
    fig, axes = plt.subplots(3, 6, figsize=(15.0, 7.2), constrained_layout=True)
    image = None

    for row, (family_label, factory) in enumerate(family_specs):
        for column, n in enumerate(range(1, 7)):
            state = factory(n, 29)
            magnitude = np.abs(np.asarray(state.rho))
            normalized = magnitude / magnitude.max()
            image = axes[row, column].imshow(
                normalized,
                origin="upper",
                cmap="magma",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].grid(False)
            if row == 0:
                dimension = 2**n
                axes[row, column].set_title(f"{n}Q\n{dimension}×{dimension}")
            if column == 0:
                axes[row, column].set_ylabel(
                    family_label,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=15,
                )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.46,
        pad=0.035,
        aspect=36,
    )
    colorbar.set_label(r"Per-panel normalized magnitude $|\rho_{ij}| / \max|\rho_{ij}|$")
    fig.suptitle("State-generation families from 1 to 6 qubits", fontsize=16)
    _save_figure(fig, "state_families_1_to_6_qubits")


def plot_three_required_families():
    n = 3
    d = 2**n
    states = (
        random_product_state(LIKE, n, seed=71),
        random_haar_state(LIKE, n, seed=71),
        random_mixed_state(LIKE, n, k=d, seed=71),
    )
    labels = (
        "Random product pure",
        "Haar-random pure",
        r"Induced mixed ($K=d$)",
    )
    colors = (ORANGE, BLUE, GREEN)
    magnitudes = [np.abs(np.asarray(state.rho)) for state in states]
    common_maximum = max(matrix.max() for matrix in magnitudes)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.2, 7.1),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.15, 0.85)},
    )
    image = None
    for column, (state, label, color, magnitude) in enumerate(
        zip(states, labels, colors, magnitudes)
    ):
        image = axes[0, column].imshow(
            magnitude,
            origin="upper",
            cmap="magma",
            vmin=0,
            vmax=common_maximum,
            interpolation="nearest",
        )
        axes[0, column].set_title(f"{chr(65 + column)}  {label}")
        axes[0, column].set_xlabel("Column basis index j")
        if column == 0:
            axes[0, column].set_ylabel("Row basis index i")
        axes[0, column].set_xticks(range(d))
        axes[0, column].set_yticks(range(d))
        axes[0, column].grid(False)

        eigenvalues = np.linalg.eigvalsh(np.asarray(state.rho)).real
        eigenvalues = np.sort(np.clip(eigenvalues, 0, None))[::-1]
        axes[1, column].bar(
            np.arange(d),
            eigenvalues,
            width=0.72,
            color=color,
            alpha=0.9,
        )
        axes[1, column].set(
            xlabel="Sorted eigenvalue index",
            ylim=(0, 1.05),
            title="Eigenvalue spectrum",
        )
        axes[1, column].set_xticks(range(d))
        if column == 0:
            axes[1, column].set_ylabel("Eigenvalue")

    colorbar = fig.colorbar(
        image,
        ax=axes[0, :],
        orientation="horizontal",
        shrink=0.56,
        pad=0.02,
        aspect=35,
    )
    colorbar.set_label(r"Density-matrix magnitude $|\rho_{ij}|$ (shared scale)")
    fig.suptitle("Three required state-generation families (n=3)", fontsize=16)
    _save_figure(fig, "three_generation_families")


def plot_pure_state_structure():
    bloch = np.asarray(
        [_bloch_vector(random_haar_state(LIKE, 1, seed=seed).ket) for seed in range(320)]
    )

    product_entropy = np.asarray(
        [
            _bipartite_entropy_bits(random_product_state(LIKE, 4, seed=1000 + seed).ket)
            for seed in range(240)
        ]
    )
    haar_entropy = np.asarray(
        [
            _bipartite_entropy_bits(random_haar_state(LIKE, 4, seed=2000 + seed).ket)
            for seed in range(240)
        ]
    )

    fig = plt.figure(figsize=(12.2, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.25))
    ax_sphere = fig.add_subplot(grid[0, 0], projection="3d")
    ax_entropy = fig.add_subplot(grid[0, 1])

    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax_sphere.plot_wireframe(x, y, z, color=GRID, linewidth=0.45, alpha=0.5)
    ax_sphere.scatter(
        bloch[:, 0], bloch[:, 1], bloch[:, 2], s=12, alpha=0.68, color=BLUE
    )
    ax_sphere.set(
        xlabel=r"$\langle X\rangle$",
        ylabel=r"$\langle Y\rangle$",
        zlabel=r"$\langle Z\rangle$",
        title="A  Bloch-sphere samples",
    )
    ax_sphere.set_box_aspect((1, 1, 1))
    ax_sphere.grid(False)

    # Deterministic vertical jitter avoids using any additional random source
    # while making the full empirical distributions visible.
    product_jitter = 0.13 * np.sin(np.arange(product_entropy.size) * 2.399)
    haar_jitter = 1 + 0.13 * np.sin(np.arange(haar_entropy.size) * 2.399)
    ax_entropy.scatter(
        product_entropy,
        product_jitter,
        s=17,
        color=ORANGE,
        alpha=0.42,
        linewidths=0,
        label="Product samples",
    )
    ax_entropy.scatter(
        haar_entropy,
        haar_jitter,
        s=17,
        color=BLUE,
        alpha=0.42,
        linewidths=0,
        label="Global Haar samples",
    )
    box = ax_entropy.boxplot(
        [product_entropy, haar_entropy],
        vert=False,
        positions=[0, 1],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": DARK, "linewidth": 2},
        whiskerprops={"color": DARK},
        capprops={"color": DARK},
    )
    box["boxes"][0].set(facecolor=ORANGE, alpha=0.38, edgecolor=ORANGE)
    box["boxes"][1].set(facecolor=BLUE, alpha=0.38, edgecolor=BLUE)
    page_value = _page_entropy_bits(4, 4)
    ax_entropy.axvline(
        page_value,
        color=PURPLE,
        linewidth=2,
        linestyle="--",
        label=f"Page mean = {page_value:.3f} bits",
    )
    ax_entropy.set(
        xlim=(-0.05, 2),
        xlabel="2|2 bipartite entanglement entropy (bits)",
        title="B  Bipartite entropy distributions (n=4)",
        ylim=(-0.48, 1.48),
    )
    ax_entropy.set_yticks([0, 1], ["Product pure", "Global Haar pure"])
    ax_entropy.legend(loc="upper left")
    fig.suptitle("Pure-state ensembles used in tomography benchmarks", fontsize=16)
    _save_figure(fig, "pure_state_ensembles")


def plot_mixed_state_purity():
    n = 3
    d = 2**n
    k_values = np.asarray([1, 2, 4, 8, 16, 32])
    repeats = 120
    sample_means = []
    confidence95 = []

    for position, k in enumerate(k_values):
        purities = np.asarray(
            [
                random_mixed_state(
                    LIKE,
                    n,
                    k=int(k),
                    seed=10_000 + 1000 * position + repetition,
                ).purity
                for repetition in range(repeats)
            ]
        )
        sample_means.append(purities.mean())
        confidence95.append(1.96 * purities.std(ddof=1) / math.sqrt(repeats))

    sample_means = np.asarray(sample_means)
    confidence95 = np.asarray(confidence95)
    k_grid = np.geomspace(1, 32, 240)
    theory = (d + k_grid) / (d * k_grid + 1)

    targets = np.linspace(1 / d, 1.0, 12)
    observed_haar = np.asarray(
        [
            random_state_with_purity(LIKE, n, target, seed=30_000 + index).purity
            for index, target in enumerate(targets)
        ]
    )
    observed_product = np.asarray(
        [
            random_state_with_purity(
                LIKE,
                n,
                target,
                seed=40_000 + index,
                base="product",
            ).purity
            for index, target in enumerate(targets)
        ]
    )
    fig, (ax_induced, ax_target) = plt.subplots(
        1, 2, figsize=(12.2, 4.9), constrained_layout=True
    )
    ax_induced.plot(k_grid, theory, color=BLUE, linewidth=2.2, label="Theory")
    ax_induced.errorbar(
        k_values,
        sample_means,
        yerr=confidence95,
        fmt="o",
        markersize=6,
        capsize=4,
        color=ORANGE,
        label="Monte Carlo mean ± 95% CI",
    )
    ax_induced.axhline(1 / d, color=DARK, linestyle=":", label=r"Minimum $1/d$")
    ax_induced.set_xscale("log", base=2)
    ax_induced.set_xticks(k_values, [str(value) for value in k_values])
    ax_induced.set(
        xlabel="Environment dimension K",
        ylabel=r"Purity $\mathrm{Tr}(\rho^2)$",
        title="A  Induced mixed-state ensemble (n=3)",
        ylim=(0.1, 1.04),
    )
    ax_induced.legend()

    ax_target.plot(
        [1 / d, 1], [1 / d, 1], color=DARK, linewidth=1.8, label="Ideal y=x"
    )
    ax_target.scatter(
        targets,
        observed_haar,
        s=45,
        marker="o",
        color=GREEN,
        label="Haar base",
        zorder=3,
    )
    ax_target.scatter(
        targets,
        observed_product,
        s=48,
        marker="x",
        linewidth=1.8,
        color=PURPLE,
        label="Product base",
        zorder=4,
    )
    ax_target.set(
        xlabel="Target purity",
        ylabel="Generated purity",
        title="B  Exact-purity calibration (n=3)",
        xlim=(0.1, 1.03),
        ylim=(0.1, 1.03),
    )
    ax_target.set_aspect("equal", adjustable="box")
    ax_target.legend(loc="lower right")

    fig.suptitle("Mixed-state purity controls", fontsize=16)
    _save_figure(fig, "mixed_state_purity")


def main():
    _configure_style()
    plot_families_one_to_six_qubits()
    plot_three_required_families()
    plot_pure_state_structure()
    plot_mixed_state_purity()
    print(f"Poster figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
