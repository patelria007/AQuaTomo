"""Command line interface for data generation, reconstruction, and benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backend import scalar
from .experiment import STATE_GENERATORS, benchmark, reconstruct_all, summarize
from .io import load_measurement_bundle, save_measurement_bundle, write_csv
from .measurements import simulate_pauli_measurements
from .metrics import fidelity


def _backend(name):
    if name == "numpy":
        return np
    if name == "cupy":
        import cupy as cp

        return cp
    if name == "jax":
        import jax.numpy as jnp

        return jnp
    raise ValueError(name)


def _generate(args):
    xp = _backend(args.backend)
    rng = np.random.default_rng(args.seed)
    generator = STATE_GENERATORS[args.state_type]
    states, datasets = [], []
    for _ in range(args.samples):
        state = generator(args.qubits, xp=xp, rng=rng)
        states.append(state)
        datasets.append(simulate_pauli_measurements(state, args.shots, rng=rng))
    save_measurement_bundle(
        args.output,
        states,
        datasets,
        {"backend": args.backend, "state_type": args.state_type, "seed": args.seed},
    )
    print(f"Saved {args.samples} states and measurement data to {args.output}")


def _reconstruct(args):
    xp = _backend(args.backend)
    states, datasets, metadata = load_measurement_bundle(args.input, xp=xp)
    rows = []
    for index, (truth, data) in enumerate(zip(states, datasets)):
        estimates = reconstruct_all(data, mle_iterations=args.mle_iterations, low_rank=args.rank)
        for method, estimate in estimates.items():
            rows.append({"sample": index, "method": method, "fidelity": scalar(fidelity(truth, estimate))})
    write_csv(args.output, rows)
    print(json.dumps({"input_metadata": metadata, "results": args.output}, indent=2))


def _benchmark(args):
    xp = _backend(args.backend)
    records = benchmark(
        qubits=args.qubits,
        shots=args.shots,
        state_types=args.state_types,
        states_per_case=args.states,
        xp=xp,
        seed=args.seed,
        mle_iterations=args.mle_iterations,
    )
    write_csv(args.output, records)
    summary_path = str(Path(args.output).with_name(Path(args.output).stem + "_summary.csv"))
    write_csv(summary_path, summarize(records))
    print(f"Wrote {len(records)} records to {args.output} and summary to {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(prog="nbqst", description="Hardware-agnostic quantum state tomography")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate states and finite-shot Pauli data")
    generate.add_argument("--qubits", type=int, default=2)
    generate.add_argument("--shots", type=int, default=1000)
    generate.add_argument("--samples", type=int, default=10)
    generate.add_argument("--state-type", choices=tuple(STATE_GENERATORS), default="haar")
    generate.add_argument("--backend", choices=("numpy", "cupy", "jax"), default="numpy")
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--output", default="data/tomography_data.npz")
    generate.set_defaults(func=_generate)

    reconstruct = sub.add_parser("reconstruct", help="Run all reconstruction/denoising engines")
    reconstruct.add_argument("input")
    reconstruct.add_argument("--backend", choices=("numpy", "cupy", "jax"), default="numpy")
    reconstruct.add_argument("--rank", type=int, default=1)
    reconstruct.add_argument("--mle-iterations", type=int, default=100)
    reconstruct.add_argument("--output", default="results/reconstruction.csv")
    reconstruct.set_defaults(func=_reconstruct)

    bench = sub.add_parser("benchmark", help="Benchmark state classes, shots, and estimators")
    bench.add_argument("--qubits", type=int, nargs="+", default=[1, 2])
    bench.add_argument("--shots", type=int, nargs="+", default=[100, 1000])
    bench.add_argument("--state-types", nargs="+", choices=tuple(STATE_GENERATORS), default=list(STATE_GENERATORS))
    bench.add_argument("--states", type=int, default=3)
    bench.add_argument("--backend", choices=("numpy", "cupy", "jax"), default="numpy")
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--mle-iterations", type=int, default=100)
    bench.add_argument("--output", default="results/benchmark.csv")
    bench.set_defaults(func=_benchmark)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

