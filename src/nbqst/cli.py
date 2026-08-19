"""Command line interface for data generation, reconstruction, and benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .backend import backend_runtime_metadata, scalar
from .experiment import METHOD_ALIASES, STATE_GENERATORS, benchmark, normalize_methods, reconstruct_all, summarize
from .io import load_measurement_bundle, save_measurement_bundle, write_csv
from .measurements import simulate_pauli_measurements
from .metrics import fidelity
from .neural import load_neural_model


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


def _load_neural_models(specifications):
    models = {}
    names = {}
    for specification in specifications:
        expected_qubits = None
        path_text = specification
        if "=" in specification:
            prefix, path_text = specification.split("=", 1)
            try:
                expected_qubits = int(prefix)
            except ValueError as error:
                raise ValueError(
                    f"Invalid neural model specification {specification!r}; use QUBITS=PATH or PATH"
                ) from error
        path = Path(path_text)
        model = load_neural_model(path)
        if expected_qubits is not None and model.n_qubits != expected_qubits:
            raise ValueError(
                f"Model {path} contains {model.n_qubits} qubits, not the declared {expected_qubits}"
            )
        if model.n_qubits in models:
            raise ValueError(f"More than one neural model was supplied for {model.n_qubits} qubits")
        models[model.n_qubits] = model
        names[model.n_qubits] = str(path)
    return models, names


def _benchmark_manifest(args, xp, *, output, methods, neural_model_names, record_count):
    environment_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "XLA_FLAGS",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
    )
    return {
        "schema_version": 2,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "output": str(output),
        "record_count": record_count,
        "backend": backend_runtime_metadata(xp),
        "host": {
            "hostname": platform.node(),
            "operating_system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count_visible": os.cpu_count(),
            "python": sys.version.replace("\n", " "),
        },
        "grid": {
            "qubits": list(args.qubits),
            "shots_per_setting": list(args.shots),
            "state_types": list(args.state_types),
            "states_per_case": args.states,
            "methods": list(methods),
            "mle_iterations": args.mle_iterations,
            "warmup_rounds": args.warmup_rounds,
            "timing_repeats": args.timing_repeats,
            "seed": args.seed,
        },
        "neural_models": {str(key): value for key, value in neural_model_names.items()},
        "timing_methodology": {
            "clock": "time.perf_counter_ns",
            "reconstruction_and_fidelity_timed_separately": True,
            "mle_includes_linear_inversion_and_physical_initialization": True,
            "neural_weights_moved_to_backend_before_warmup": True,
            "accelerator_synchronization": backend_runtime_metadata(xp)["synchronization"],
            "compilation_policy": (
                "warm-up excluded from recorded samples"
                if args.warmup_rounds
                else "no warm-up; first recorded sample may include compilation"
            ),
            "measurement_generation": "timed once per shared dataset and repeated in each method row",
        },
        "environment": {name: os.environ.get(name, "unset") for name in environment_names},
    }


def _benchmark(args):
    xp = _backend(args.backend)
    neural_models, neural_model_names = _load_neural_models(args.neural_model)
    methods = normalize_methods(
        args.methods
        if args.methods is not None
        else (
            ("linear_inversion", "maximum_likelihood", "neural_network")
            if neural_models
            else ("linear_inversion", "maximum_likelihood")
        )
    )
    records = benchmark(
        qubits=args.qubits,
        shots=args.shots,
        state_types=args.state_types,
        states_per_case=args.states,
        xp=xp,
        seed=args.seed,
        mle_iterations=args.mle_iterations,
        methods=methods,
        neural_models=neural_models,
        neural_model_names=neural_model_names,
        warmup_rounds=args.warmup_rounds,
        timing_repeats=args.timing_repeats,
    )
    output = Path(args.output or f"results/benchmark_{args.backend}.csv")
    write_csv(output, records)
    summary_path = output.with_name(output.stem + "_summary.csv")
    write_csv(summary_path, summarize(records))
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(
            _benchmark_manifest(
                args,
                xp,
                output=output,
                methods=methods,
                neural_model_names=neural_model_names,
                record_count=len(records),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(records)} synchronized timing records to {output}, "
        f"summary to {summary_path}, and manifest to {manifest_path}"
    )


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

    bench = sub.add_parser("benchmark", help="Benchmark synchronized LI, MLE, and neural timing across a scaling grid")
    bench.add_argument("--qubits", type=int, nargs="+", default=[1, 2])
    bench.add_argument("--shots", type=int, nargs="+", default=[100, 1000])
    bench.add_argument("--state-types", nargs="+", choices=tuple(STATE_GENERATORS), default=list(STATE_GENERATORS))
    bench.add_argument("--states", type=int, default=3)
    bench.add_argument("--backend", choices=("numpy", "cupy", "jax"), default="numpy")
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--mle-iterations", type=int, default=100)
    bench.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_ALIASES),
        default=None,
        help="Methods to time. Defaults to LI+MLE, or LI+MLE+NN when a neural model is supplied.",
    )
    bench.add_argument(
        "--neural-model",
        action="append",
        default=[],
        metavar="[QUBITS=]PATH",
        help="Repeat once per qubit size, e.g. --neural-model 2=results/model_2q.npz",
    )
    bench.add_argument("--warmup-rounds", type=int, default=1)
    bench.add_argument("--timing-repeats", type=int, default=1)
    bench.add_argument(
        "--output",
        default=None,
        help="Detailed CSV path. Defaults to results/benchmark_<backend>.csv to prevent backend runs overwriting each other.",
    )
    bench.set_defaults(func=_benchmark)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
