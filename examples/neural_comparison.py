"""Train and compare Linear Inversion, MLE, and neural reconstruction.

This is a compact implementation-validation experiment, not a reproduction of
the much larger training study in arXiv:2206.06736.  The three estimators see
identical finite-shot local-Pauli data.  Neural training time is recorded
separately and never folded into per-state inference time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from nbqst.io import write_csv
from nbqst.measurements import simulate_pauli_measurements
from nbqst.metrics import fidelity, hilbert_schmidt_distance, minimum_eigenvalue
from nbqst.neural import neural_state_reconstruction, save_neural_model, train_neural_reconstructor
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli
from nbqst.states import haar_random_pure, random_mixed_state, random_product_state


GENERATORS = {
    "product": random_product_state,
    "haar": haar_random_pure,
    "mixed": random_mixed_state,
}


def generate_dataset(n_qubits, samples, shots, rng):
    states, datasets, labels = [], [], []
    names = tuple(GENERATORS)
    for index in range(samples):
        label = names[index % len(names)]
        state = GENERATORS[label](n_qubits, rng=rng)
        states.append(state)
        datasets.append(simulate_pauli_measurements(state, shots, rng=rng))
        labels.append(label)
    return states, datasets, labels


def summarize(rows):
    summary = []
    for method in ("linear_inversion", "maximum_likelihood", "neural_network"):
        selected = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "samples": len(selected),
                "mean_hs_distance": np.mean([row["hs_distance"] for row in selected]),
                "mean_fidelity": np.mean([row["fidelity"] for row in selected]),
                "physical_fraction": np.mean([row["minimum_eigenvalue"] >= -1e-10 for row in selected]),
                "mean_inference_seconds": np.mean([row["inference_seconds"] for row in selected]),
            }
        )
    return summary


def run(args):
    rng = np.random.default_rng(args.seed)
    train_states, train_data, _ = generate_dataset(args.qubits, args.training_samples, args.shots, rng)
    start = time.perf_counter()
    model, history = train_neural_reconstructor(
        train_states,
        train_data,
        hidden_layers=tuple(args.hidden_layers),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed + 1,
        return_history=True,
    )
    training_seconds = time.perf_counter() - start

    test_states, test_data, labels = generate_dataset(args.qubits, args.test_samples, args.shots, rng)
    rows = []
    for index, (truth, data, label) in enumerate(zip(test_states, test_data, labels)):
        estimates = {}
        timings = {}
        start = time.perf_counter()
        estimates["linear_inversion"] = linear_inversion_pauli(data)
        timings["linear_inversion"] = time.perf_counter() - start
        start = time.perf_counter()
        estimates["maximum_likelihood"] = factorized_mle(
            data,
            max_iter=args.mle_iterations,
        )
        timings["maximum_likelihood"] = time.perf_counter() - start
        start = time.perf_counter()
        estimates["neural_network"] = neural_state_reconstruction(data, model)
        timings["neural_network"] = time.perf_counter() - start
        for method, estimate in estimates.items():
            rows.append(
                {
                    "sample": index,
                    "state_type": label,
                    "shots_per_setting": args.shots,
                    "method": method,
                    "hs_distance": float(hilbert_schmidt_distance(truth, estimate)),
                    "fidelity": float(fidelity(truth, estimate)),
                    "minimum_eigenvalue": float(minimum_eigenvalue(estimate)),
                    "inference_seconds": timings[method],
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    write_csv(output.with_name(output.stem + "_summary.csv"), summarize(rows))
    write_csv(output.with_name(output.stem + "_training_history.csv"), history)
    save_neural_model(output.with_name(output.stem + "_model.npz"), model)
    metadata = [
        {
            "n_qubits": args.qubits,
            "shots_per_setting": args.shots,
            "training_samples": args.training_samples,
            "test_samples": args.test_samples,
            "hidden_layers": "x".join(str(value) for value in args.hidden_layers),
            "epochs_completed": len(history),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "mle_iterations": args.mle_iterations,
            "seed": args.seed,
            "training_seconds": training_seconds,
        }
    ]
    write_csv(output.with_name(output.stem + "_configuration.csv"), metadata)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--qubits", type=int, default=2)
    result.add_argument("--shots", type=int, default=500)
    result.add_argument("--training-samples", type=int, default=600)
    result.add_argument("--test-samples", type=int, default=60)
    result.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 64])
    result.add_argument("--epochs", type=int, default=250)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--patience", type=int, default=40)
    result.add_argument("--mle-iterations", type=int, default=60)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--output", default="results/neural_comparison.csv")
    return result


if __name__ == "__main__":
    run(parser().parse_args())

