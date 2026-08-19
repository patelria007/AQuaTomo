"""Reproducible end-to-end experiments and CSV-friendly benchmark records."""

from __future__ import annotations

import numpy as np

from .backend import backend_runtime_metadata, scalar, synchronize, timed_call
from .denoise import depolarizing_shrinkage, low_rank_projection, project_density_matrix
from .measurements import simulate_pauli_measurements
from .metrics import fidelity, hilbert_schmidt_distance, minimum_eigenvalue, purity
from .neural import NeuralTomographyModel, neural_model_to_backend, neural_state_reconstruction
from .reconstruction import factorized_mle, linear_inversion_pauli
from .states import haar_random_pure, random_mixed_state, random_product_state


STATE_GENERATORS = {
    "product": random_product_state,
    "haar": haar_random_pure,
    "mixed": random_mixed_state,
}

TOMOGRAPHY_METHODS = ("linear_inversion", "maximum_likelihood", "neural_network")
METHOD_ALIASES = {
    "li": "linear_inversion",
    "linear": "linear_inversion",
    "linear_inversion": "linear_inversion",
    "mle": "maximum_likelihood",
    "maximum_likelihood": "maximum_likelihood",
    "nn": "neural_network",
    "neural": "neural_network",
    "neural_network": "neural_network",
}


def reconstruct_all(data, *, mle_iterations=100, low_rank=1, shrinkage_alpha=0.9):
    linear = linear_inversion_pauli(data)
    projected = project_density_matrix(linear)
    return {
        "linear": linear,
        "projected": projected,
        "low_rank": low_rank_projection(linear, min(low_rank, linear.shape[-1])),
        "shrinkage": depolarizing_shrinkage(linear, shrinkage_alpha),
        "mle": factorized_mle(data, initial=projected, max_iter=mle_iterations),
    }


def normalize_methods(methods):
    normalized = []
    for method in methods:
        try:
            canonical = METHOD_ALIASES[method.lower().replace("-", "_")]
        except KeyError as error:
            raise ValueError(f"Unknown method {method!r}; choose from {TOMOGRAPHY_METHODS}") from error
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("At least one reconstruction method is required")
    return tuple(normalized)


def _neural_parameter_count(model: NeuralTomographyModel) -> int:
    return int(sum(weight.size + bias.size for weight, bias in zip(model.weights, model.biases)))


def _measurement_storage_bytes(data) -> int:
    total = 0
    for counts in data.counts.values():
        itemsize = getattr(getattr(counts, "dtype", None), "itemsize", 0)
        total += int(counts.size) * int(itemsize)
    return total


def _estimate(method, data, *, mle_iterations, neural_model=None):
    if method == "linear_inversion":
        return linear_inversion_pauli(data), 0
    if method == "maximum_likelihood":
        # Deliberately include LI + physical initialization in the MLE timer.
        # This is the end-to-end cost a user pays for a standalone MLE call.
        estimate, history = factorized_mle(data, max_iter=mle_iterations, return_history=True)
        return estimate, max(0, len(history) - 1)
    if method == "neural_network":
        if neural_model is None:
            raise ValueError(f"A trained neural model is required for {data.n_qubits} qubits")
        return neural_state_reconstruction(data, neural_model), 0
    raise ValueError(method)


def benchmark(
    *,
    qubits=(1, 2),
    shots=(100, 1000),
    state_types=("product", "haar", "mixed"),
    states_per_case=3,
    xp=np,
    seed=7,
    mle_iterations=100,
    methods=("linear_inversion", "maximum_likelihood"),
    neural_models=None,
    neural_model_names=None,
    warmup_rounds=1,
    timing_repeats=1,
):
    """Time reconstruction and fidelity separately across a scaling grid.

    All timed regions are synchronized.  MLE is timed end-to-end, including
    its LI/physical initialization.  Neural weights are moved to the selected
    backend once before warm-up and timing, so resident inference is measured.
    Measurement generation is timed once per shared dataset and copied into
    each method record; it is never hidden inside one estimator's timer.
    """

    if states_per_case < 1 or warmup_rounds < 0 or timing_repeats < 1:
        raise ValueError("states_per_case and timing_repeats must be positive; warmup_rounds must be nonnegative")
    if mle_iterations < 1:
        raise ValueError("mle_iterations must be positive")
    methods = normalize_methods(methods)
    unknown_state_types = set(state_types) - set(STATE_GENERATORS)
    if unknown_state_types:
        raise ValueError(f"Unknown state types: {sorted(unknown_state_types)}")

    neural_models = dict(neural_models or {})
    neural_model_names = dict(neural_model_names or {})
    if "neural_network" in methods:
        missing = sorted(set(qubits) - set(neural_models))
        if missing:
            raise ValueError(
                "Neural timing requires one trained model per qubit count; "
                f"missing models for {missing}"
            )
        neural_models = {
            n_qubits: neural_model_to_backend(model, xp)
            for n_qubits, model in neural_models.items()
        }
        synchronize(
            tuple(
                value
                for model in neural_models.values()
                for value in (*model.weights, *model.biases)
            ),
            xp=xp,
        )

    generator = np.random.default_rng(seed)
    runtime = backend_runtime_metadata(xp)
    records = []
    warmed = set()
    for n_qubits in qubits:
        for state_type in state_types:
            state_generator = STATE_GENERATORS[state_type]
            for state_index in range(states_per_case):
                rho_true, state_generation_seconds = timed_call(
                    lambda: state_generator(n_qubits, xp=xp, rng=generator),
                    xp=xp,
                )
                for shot_count in shots:
                    data, measurement_seconds = timed_call(
                        lambda: simulate_pauli_measurements(rho_true, shot_count, rng=generator),
                        xp=xp,
                        synchronize_before=rho_true,
                    )
                    measurement_bytes = _measurement_storage_bytes(data)
                    for method in methods:
                        model = neural_models.get(n_qubits)
                        warmup_key = (n_qubits, method)
                        if warmup_key not in warmed:
                            for _ in range(warmup_rounds):
                                warm_result, _ = timed_call(
                                    lambda: _estimate(
                                        method,
                                        data,
                                        mle_iterations=mle_iterations,
                                        neural_model=model,
                                    ),
                                    xp=xp,
                                    synchronize_before=data.counts,
                                )
                                warm_estimate = warm_result[0]
                                _, _ = timed_call(
                                    lambda: fidelity(rho_true, warm_estimate),
                                    xp=xp,
                                    synchronize_before=(rho_true, warm_estimate),
                                )
                            warmed.add(warmup_key)

                        for timing_repeat in range(timing_repeats):
                            reconstruction_result, reconstruction_seconds = timed_call(
                                lambda: _estimate(
                                    method,
                                    data,
                                    mle_iterations=mle_iterations,
                                    neural_model=model,
                                ),
                                xp=xp,
                                synchronize_before=data.counts,
                            )
                            estimate, mle_iterations_completed = reconstruction_result
                            fidelity_value, fidelity_seconds = timed_call(
                                lambda: fidelity(rho_true, estimate),
                                xp=xp,
                                synchronize_before=(rho_true, estimate),
                            )
                            method_total_seconds = reconstruction_seconds + fidelity_seconds
                            end_to_end_seconds = state_generation_seconds + measurement_seconds + method_total_seconds
                            neural_parameters = _neural_parameter_count(model) if method == "neural_network" else 0
                            minimum = scalar(minimum_eigenvalue(estimate))
                            trace_value = scalar(xp.real(xp.trace(estimate)))
                            is_physical = minimum >= -1e-10 and abs(trace_value - 1.0) <= 1e-8
                            row = {
                                **runtime,
                                "benchmark_schema_version": 2,
                                "seed": seed,
                                "n_qubits": n_qubits,
                                "dimension": 2**n_qubits,
                                "state_type": state_type,
                                "state_index": state_index,
                                "shots_per_setting": shot_count,
                                "settings": 3**n_qubits,
                                "outcomes_per_setting": 2**n_qubits,
                                "frequency_values": 6**n_qubits,
                                "total_shots": (3**n_qubits) * shot_count,
                                "measurement_storage_bytes": measurement_bytes,
                                "method": method,
                                "timing_repeat": timing_repeat,
                                "warmup_rounds": warmup_rounds,
                                "timing_repeats": timing_repeats,
                                "mle_iterations_requested": mle_iterations if method == "maximum_likelihood" else 0,
                                "mle_iterations_completed": mle_iterations_completed,
                                "neural_parameter_count": neural_parameters,
                                "neural_model": neural_model_names.get(n_qubits, "in_memory") if method == "neural_network" else "not_applicable",
                                "fidelity": scalar(fidelity_value),
                                "fidelity_interpretable": is_physical,
                                "hs_distance": scalar(hilbert_schmidt_distance(rho_true, estimate)),
                                "estimated_purity": scalar(purity(estimate)),
                                "minimum_eigenvalue": minimum,
                                "trace": trace_value,
                                "is_physical": is_physical,
                                "state_generation_seconds": state_generation_seconds,
                                "measurement_seconds": measurement_seconds,
                                "reconstruction_seconds": reconstruction_seconds,
                                "fidelity_seconds": fidelity_seconds,
                                "method_total_seconds": method_total_seconds,
                                "end_to_end_seconds": end_to_end_seconds,
                            }
                            records.append(row)
    return records


def summarize(records):
    grouped = {}
    for row in records:
        key = (
            row.get("backend", "unknown"),
            row.get("device_name", "unknown"),
            row["n_qubits"],
            row["state_type"],
            row["shots_per_setting"],
            row["method"],
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, rows in sorted(grouped.items()):
        reconstruction = np.asarray([r["reconstruction_seconds"] for r in rows], dtype=float)
        fidelity_times = np.asarray([r.get("fidelity_seconds", 0.0) for r in rows], dtype=float)
        totals = np.asarray([r.get("method_total_seconds", r["reconstruction_seconds"]) for r in rows], dtype=float)
        end_to_end = np.asarray([r.get("end_to_end_seconds", r["reconstruction_seconds"]) for r in rows], dtype=float)
        state_indices = {r.get("state_index", index) for index, r in enumerate(rows)}
        interpretable = [r["fidelity"] for r in rows if r.get("fidelity_interpretable", True)]
        physical_fraction = float(np.mean([bool(r.get("is_physical", True)) for r in rows]))
        summary.append(
            {
                "backend": key[0],
                "device_name": key[1],
                "n_qubits": key[2],
                "state_type": key[3],
                "shots_per_setting": key[4],
                "method": key[5],
                "mean_fidelity": sum(r["fidelity"] for r in rows) / len(rows),
                "mean_interpretable_fidelity": float(np.mean(interpretable)) if interpretable else "not_available",
                "physical_fraction": physical_fraction,
                "mean_hs_distance": sum(r["hs_distance"] for r in rows) / len(rows),
                "mean_reconstruction_seconds": float(np.mean(reconstruction)),
                "median_reconstruction_seconds": float(np.median(reconstruction)),
                "std_reconstruction_seconds": float(np.std(reconstruction, ddof=1)) if len(rows) > 1 else 0.0,
                "p95_reconstruction_seconds": float(np.percentile(reconstruction, 95)),
                "mean_fidelity_seconds": float(np.mean(fidelity_times)),
                "median_fidelity_seconds": float(np.median(fidelity_times)),
                "p95_fidelity_seconds": float(np.percentile(fidelity_times, 95)),
                "mean_method_total_seconds": float(np.mean(totals)),
                "median_method_total_seconds": float(np.median(totals)),
                "p95_method_total_seconds": float(np.percentile(totals, 95)),
                "mean_end_to_end_seconds": float(np.mean(end_to_end)),
                "median_end_to_end_seconds": float(np.median(end_to_end)),
                "p95_end_to_end_seconds": float(np.percentile(end_to_end, 95)),
                "warmup_rounds": max(r.get("warmup_rounds", 0) for r in rows),
                "unique_states": len(state_indices),
                "timed_samples": len(rows),
                "replicates": len(rows),
            }
        )
    return summary
