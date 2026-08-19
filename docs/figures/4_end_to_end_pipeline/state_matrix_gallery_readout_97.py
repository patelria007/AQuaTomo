"""Plot the four-qubit state-matrix gallery with uniform 97% readout fidelity.

Run from the repository root with::

    python verification/4_end_to_end_pipeline/state_matrix_gallery_readout_97.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from state_matrix_gallery import (
    DEFAULT_MLE_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_SHOTS,
    build_gallery_data,
    plot_gallery,
    print_report,
)


READOUT_FIDELITY = 0.97
DEFAULT_OUTPUT = Path(__file__).with_name("state_matrix_gallery_readout_97.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mle-iterations", type=int, default=DEFAULT_MLE_ITERATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_gallery_data(
        args.shots,
        args.seed,
        args.mle_iterations,
        readout_fidelity_0=READOUT_FIDELITY,
        readout_fidelity_1=READOUT_FIDELITY,
    )
    print_report(
        rows,
        args.shots,
        readout_fidelity_0=READOUT_FIDELITY,
        readout_fidelity_1=READOUT_FIDELITY,
    )
    plot_gallery(
        rows,
        args.output,
        args.shots,
        show=args.show,
        readout_fidelity_0=READOUT_FIDELITY,
        readout_fidelity_1=READOUT_FIDELITY,
    )
    print(f"\nSaved figure: {args.output}")


if __name__ == "__main__":
    main()
