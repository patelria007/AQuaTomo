"""Monte Carlo distribution checks for the four-qubit state generators.

Run from the repository root with::

    python verification/state_generation/state_distribution_verification.py

The checks are deliberately complementary to ``state_generation_verification``:
that script checks individual density matrices, while this script checks the
probability laws obtained from many independently generated states.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
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
MIXED_RANKS = (4, DIMENSION)
DEFAULT_SAMPLES = 2000
DEFAULT_SEED = 20260818
DEFAULT_OUTPUT = Path(__file__).with_name(
    "four_qubit_distribution_verification.png"
)
KS_ALPHA = 0.01


@dataclass(frozen=True)
class Check:
    """One scalar acceptance check shown in the terminal report."""

    name: str
    observed: float
    limit: float
    context: str

    @property
    def passed(self) -> bool:
        return self.observed <= self.limit


def partial_trace(rho: np.ndarray, keep: tuple[int, ...]) -> np.ndarray:
    """Trace out all qubits except ``keep`` (most-significant-qubit first)."""
    keep = tuple(keep)
    if not keep or len(set(keep)) != len(keep):
        raise ValueError("keep must contain distinct qubit indices")
    if any(index < 0 or index >= N_QUBITS for index in keep):
        raise ValueError(f"qubit indices must be in [0, {N_QUBITS - 1}]")

    traced = tuple(index for index in range(N_QUBITS) if index not in keep)
    tensor = np.asarray(rho).reshape((2,) * (2 * N_QUBITS))
    permutation = (
        list(keep)
        + list(traced)
        + [N_QUBITS + index for index in keep]
        + [N_QUBITS + index for index in traced]
    )
    tensor = np.transpose(tensor, permutation)
    kept_dimension = 2 ** len(keep)
    traced_dimension = 2 ** len(traced)
    tensor = tensor.reshape(
        kept_dimension,
        traced_dimension,
        kept_dimension,
        traced_dimension,
    )
    return np.trace(tensor, axis1=1, axis2=3)


def purity(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """Return (<X>, <Y>, <Z>) for a one-qubit density matrix."""
    return np.asarray(
        [
            2.0 * rho[0, 1].real,
            -2.0 * rho[0, 1].imag,
            (rho[0, 0] - rho[1, 1]).real,
        ]
    )


def kolmogorov_smirnov_statistic(values: np.ndarray, cdf) -> float:
    """One-sample two-sided KS statistic without a SciPy dependency."""
    ordered = np.sort(np.asarray(values, dtype=float))
    probabilities = np.clip(np.asarray(cdf(ordered), dtype=float), 0.0, 1.0)
    count = ordered.size
    upper = np.arange(1, count + 1) / count
    lower = np.arange(count) / count
    return float(max(np.max(upper - probabilities), np.max(probabilities - lower)))


def ks_critical_value(sample_count: int, alpha: float = KS_ALPHA) -> float:
    """Asymptotic two-sided KS critical value at significance ``alpha``."""
    return math.sqrt(-0.5 * math.log(alpha / 2.0)) / math.sqrt(sample_count)


def induced_mean_purity(rank: int) -> float:
    """E[Tr(rho^2)] for a d-by-rank complex Ginibre induced state."""
    return (DIMENSION + rank) / (DIMENSION * rank + 1)


def collect_samples(sample_count: int, seed: int) -> dict[str, object]:
    """Generate independent ensembles and collect sufficient statistics."""
    product_rng = np.random.default_rng(seed)
    haar_rng = np.random.default_rng(seed + 1)
    mixed_rngs = {
        rank: np.random.default_rng(seed + 100 + rank) for rank in MIXED_RANKS
    }

    product_bloch = np.empty((sample_count * N_QUBITS, 3))
    product_bipartite_purity = np.empty(sample_count)
    haar_basis_probability = np.empty(sample_count)
    haar_bipartite_purity = np.empty(sample_count)
    mixed_purities = {
        rank: np.empty(sample_count) for rank in MIXED_RANKS
    }

    state_sums = {
        "product": np.zeros((DIMENSION, DIMENSION), dtype=complex),
        "haar": np.zeros((DIMENSION, DIMENSION), dtype=complex),
        **{
            f"mixed_rank_{rank}": np.zeros(
                (DIMENSION, DIMENSION), dtype=complex
            )
            for rank in MIXED_RANKS
        },
    }

    for sample in range(sample_count):
        product = np.asarray(
            random_product_state(N_QUBITS, rng=product_rng)
        )
        haar = np.asarray(haar_random_pure(N_QUBITS, rng=haar_rng))
        state_sums["product"] += product
        state_sums["haar"] += haar

        start = sample * N_QUBITS
        for qubit in range(N_QUBITS):
            reduced = partial_trace(product, (qubit,))
            product_bloch[start + qubit] = bloch_vector(reduced)

        product_bipartite_purity[sample] = purity(
            partial_trace(product, (0, 1))
        )
        haar_basis_probability[sample] = haar[0, 0].real
        haar_bipartite_purity[sample] = purity(partial_trace(haar, (0, 1)))

        for rank in MIXED_RANKS:
            mixed = np.asarray(
                random_mixed_state(
                    N_QUBITS,
                    rank=rank,
                    rng=mixed_rngs[rank],
                )
            )
            state_sums[f"mixed_rank_{rank}"] += mixed
            mixed_purities[rank][sample] = purity(mixed)

    average_states = {
        name: total / sample_count for name, total in state_sums.items()
    }
    return {
        "product_bloch": product_bloch,
        "product_bipartite_purity": product_bipartite_purity,
        "haar_basis_probability": haar_basis_probability,
        "haar_bipartite_purity": haar_bipartite_purity,
        "mixed_purities": mixed_purities,
        "average_states": average_states,
    }


def mean_error_check(
    name: str,
    values: np.ndarray,
    expected: float,
) -> Check:
    """Compare an empirical mean with theory using a five-standard-error band."""
    sample_error = abs(float(np.mean(values)) - expected)
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    limit = max(5.0 * standard_error, 5e-4)
    return Check(
        name=name,
        observed=sample_error,
        limit=limit,
        context=f"mean={np.mean(values):.6f}, theory={expected:.6f}",
    )


def average_state_check(
    name: str,
    average_state: np.ndarray,
    theoretical_purity: float,
    sample_count: int,
) -> Check:
    """Check that the ensemble mean approaches the maximally mixed state."""
    target = np.eye(DIMENSION) / DIMENSION
    error = float(np.linalg.norm(average_state - target, ord="fro"))
    rms_scale = math.sqrt(
        max(theoretical_purity - 1.0 / DIMENSION, 0.0) / sample_count
    )
    return Check(
        name=name,
        observed=error,
        limit=5.0 * rms_scale,
        context=r"Frobenius error from I/d",
    )


def build_checks(data: dict[str, object], sample_count: int) -> tuple[Check, ...]:
    """Build distribution-level tests with explicit acceptance thresholds."""
    bloch = data["product_bloch"]
    bloch_count = bloch.shape[0]
    bloch_z = bloch[:, 2]
    z_ks = kolmogorov_smirnov_statistic(
        bloch_z,
        lambda values: (values + 1.0) / 2.0,
    )
    bloch_mean_error = float(np.max(np.abs(np.mean(bloch, axis=0))))
    bloch_second_moment = bloch.T @ bloch / bloch_count
    isotropy_error = float(
        np.max(np.abs(bloch_second_moment - np.eye(3) / 3.0))
    )
    radius_error = float(
        np.max(np.abs(np.linalg.norm(bloch, axis=1) - 1.0))
    )

    haar_probability = data["haar_basis_probability"]
    haar_ks = kolmogorov_smirnov_statistic(
        haar_probability,
        lambda values: 1.0 - (1.0 - values) ** (DIMENSION - 1),
    )

    checks = [
        Check(
            "Product Bloch-z KS",
            z_ks,
            ks_critical_value(bloch_count),
            "uniform on [-1, 1], alpha=0.01",
        ),
        Check(
            "Product Bloch mean",
            bloch_mean_error,
            5.0 * math.sqrt(1.0 / (3.0 * bloch_count)),
            "max |mean(X,Y,Z)|, target=0",
        ),
        Check(
            "Product Bloch isotropy",
            isotropy_error,
            5.0 * math.sqrt(4.0 / (45.0 * bloch_count)),
            "max second-moment error from I/3",
        ),
        Check(
            "Product Bloch radius",
            radius_error,
            1e-10,
            "local pure states lie on unit sphere",
        ),
        Check(
            "Haar basis-probability KS",
            haar_ks,
            ks_critical_value(sample_count),
            f"Beta(1,{DIMENSION - 1}), alpha=0.01",
        ),
        mean_error_check(
            "Haar 2|2 purity mean",
            data["haar_bipartite_purity"],
            (4 + 4) / (4 * 4 + 1),
        ),
        mean_error_check(
            "Product 2|2 purity",
            data["product_bipartite_purity"],
            1.0,
        ),
    ]

    for rank in MIXED_RANKS:
        checks.append(
            mean_error_check(
                f"Mixed rank-{rank} purity mean",
                data["mixed_purities"][rank],
                induced_mean_purity(rank),
            )
        )

    average_states = data["average_states"]
    checks.extend(
        [
            average_state_check(
                "Product ensemble mean",
                average_states["product"],
                1.0,
                sample_count,
            ),
            average_state_check(
                "Haar ensemble mean",
                average_states["haar"],
                1.0,
                sample_count,
            ),
            *[
                average_state_check(
                    f"Mixed rank-{rank} ensemble mean",
                    average_states[f"mixed_rank_{rank}"],
                    induced_mean_purity(rank),
                    sample_count,
                )
                for rank in MIXED_RANKS
            ],
        ]
    )
    return tuple(checks)


def print_report(checks: tuple[Check, ...], sample_count: int) -> None:
    print(f"Four-qubit distribution verification with {sample_count} samples")
    print(
        f"{'check':<38} {'observed':>11} {'limit':>11} "
        f"{'result':>8}  context"
    )
    print("-" * 105)
    for check in checks:
        result = "PASS" if check.passed else "FAIL"
        print(
            f"{check.name:<38} {check.observed:11.4e} "
            f"{check.limit:11.4e} {result:>8}  {check.context}"
        )


def plot_distributions(
    data: dict[str, object],
    sample_count: int,
    output: Path,
    *,
    show: bool = False,
) -> None:
    """Plot empirical distributions against their analytic predictions."""
    blue = "#2f6da9"
    orange = "#d66b32"
    green = "#3b8b6e"
    purple = "#7656a5"
    dark = "#30363d"

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)

    bloch_z = data["product_bloch"][:, 2]
    axes[0, 0].hist(
        bloch_z,
        bins=28,
        range=(-1.0, 1.0),
        density=True,
        color=blue,
        alpha=0.78,
        label="empirical",
    )
    axes[0, 0].axhline(0.5, color=dark, linewidth=2, label="uniform theory")
    axes[0, 0].set(
        xlabel=r"local Bloch coordinate $\langle Z\rangle$",
        ylabel="probability density",
        title="A  Product-state local Haar measure",
        xlim=(-1.0, 1.0),
    )
    axes[0, 0].legend()

    probabilities = data["haar_basis_probability"]
    probability_grid = np.linspace(0.0, 0.45, 400)
    beta_density = (DIMENSION - 1) * (1.0 - probability_grid) ** (DIMENSION - 2)
    axes[0, 1].hist(
        probabilities,
        bins=30,
        range=(0.0, 0.45),
        density=True,
        color=orange,
        alpha=0.75,
        label="empirical",
    )
    axes[0, 1].plot(
        probability_grid,
        beta_density,
        color=dark,
        linewidth=2,
        label=rf"Beta$(1,{DIMENSION - 1})$ theory",
    )
    axes[0, 1].set(
        xlabel=r"basis probability $p_0=\langle 0000|\rho|0000\rangle$",
        ylabel="probability density",
        title="B  Global Haar basis population",
        xlim=(0.0, 0.45),
    )
    axes[0, 1].legend()

    mixed_colors = {4: green, DIMENSION: purple}
    for rank in MIXED_RANKS:
        values = data["mixed_purities"][rank]
        axes[1, 0].hist(
            values,
            bins=28,
            density=True,
            alpha=0.58,
            color=mixed_colors[rank],
            label=f"rank={rank} empirical",
        )
        expected = induced_mean_purity(rank)
        axes[1, 0].axvline(
            expected,
            color=mixed_colors[rank],
            linewidth=2,
            linestyle="--",
            label=f"rank={rank} theory mean={expected:.4f}",
        )
    axes[1, 0].set(
        xlabel=r"global purity $\mathrm{Tr}(\rho^2)$",
        ylabel="probability density",
        title="C  Ginibre/Wishart mixed states",
    )
    axes[1, 0].legend()

    haar_bipartite = data["haar_bipartite_purity"]
    page_purity = (4 + 4) / (4 * 4 + 1)
    axes[1, 1].hist(
        haar_bipartite,
        bins=28,
        density=True,
        color=blue,
        alpha=0.75,
        label="global Haar empirical",
    )
    axes[1, 1].axvline(
        page_purity,
        color=dark,
        linewidth=2,
        label=f"Haar theory mean={page_purity:.4f}",
    )
    axes[1, 1].axvline(
        1.0,
        color=orange,
        linewidth=2,
        linestyle="--",
        label="product state=1",
    )
    axes[1, 1].set(
        xlabel=r"2|2 reduced purity $\mathrm{Tr}(\rho_{01}^2)$",
        ylabel="probability density",
        title="D  Pure-state entanglement structure",
        xlim=(0.35, 1.03),
    )
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"Four-qubit state-generator distribution checks (N={sample_count})",
        fontsize=16,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"independent samples per ensemble (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"base random seed (default: {DEFAULT_SEED})",
    )
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
    args = parser.parse_args()
    if args.samples < 200:
        parser.error("--samples must be at least 200 for the asymptotic tests")
    return args


def main() -> None:
    args = parse_args()
    data = collect_samples(args.samples, args.seed)
    checks = build_checks(data, args.samples)
    print_report(checks, args.samples)
    plot_distributions(data, args.samples, args.output, show=args.show)
    print(f"\nFigure written to {args.output.resolve()}")

    failures = [check.name for check in checks if not check.passed]
    if failures:
        raise SystemExit("Distribution checks failed: " + ", ".join(failures))
    print("All distribution checks passed.")


if __name__ == "__main__":
    main()
