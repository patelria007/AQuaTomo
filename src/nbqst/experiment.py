"""Reproducible end-to-end experiments and CSV-friendly benchmark records."""

from __future__ import annotations

import time

import numpy as np

from .backend import backend_name, scalar
from .denoise import depolarizing_shrinkage, low_rank_projection, project_density_matrix
from .measurements import simulate_pauli_measurements
from .metrics import fidelity, hilbert_schmidt_distance, minimum_eigenvalue, purity
from .reconstruction import factorized_mle, linear_inversion_pauli
from .states import haar_random_pure, random_mixed_state, random_product_state


STATE_GENERATORS = {
    "product": random_product_state,
    "haar": haar_random_pure,
    "mixed": random_mixed_state,
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


def benchmark(
    *,
    qubits=(1, 2),
    shots=(100, 1000),
    state_types=("product", "haar", "mixed"),
    states_per_case=3,
    xp=np,
    seed=7,
    mle_iterations=100,
):
    generator = np.random.default_rng(seed)
    records = []
    for n_qubits in qubits:
        for state_type in state_types:
            state_generator = STATE_GENERATORS[state_type]
            for state_index in range(states_per_case):
                rho_true = state_generator(n_qubits, xp=xp, rng=generator)
                for shot_count in shots:
                    data = simulate_pauli_measurements(rho_true, shot_count, rng=generator)
                    estimates = {}
                    timings = {}
                    start = time.perf_counter()
                    linear = linear_inversion_pauli(data)
                    timings["linear"] = time.perf_counter() - start
                    estimates["linear"] = linear
                    start = time.perf_counter()
                    estimates["projected"] = project_density_matrix(linear)
                    timings["projected"] = time.perf_counter() - start
                    start = time.perf_counter()
                    estimates["low_rank"] = low_rank_projection(linear, 1)
                    timings["low_rank"] = time.perf_counter() - start
                    start = time.perf_counter()
                    estimates["shrinkage"] = depolarizing_shrinkage(linear, 0.9)
                    timings["shrinkage"] = time.perf_counter() - start
                    start = time.perf_counter()
                    estimates["mle"] = factorized_mle(
                        data, initial=estimates["projected"], max_iter=mle_iterations
                    )
                    timings["mle"] = time.perf_counter() - start
                    for method, estimate in estimates.items():
                        records.append(
                            {
                                "backend": backend_name(xp),
                                "n_qubits": n_qubits,
                                "state_type": state_type,
                                "state_index": state_index,
                                "shots_per_setting": shot_count,
                                "settings": 3**n_qubits,
                                "method": method,
                                "fidelity": scalar(fidelity(rho_true, estimate)),
                                "hs_distance": scalar(hilbert_schmidt_distance(rho_true, estimate)),
                                "estimated_purity": scalar(purity(estimate)),
                                "minimum_eigenvalue": scalar(minimum_eigenvalue(estimate)),
                                "reconstruction_seconds": timings[method],
                            }
                        )
    return records


def summarize(records):
    grouped = {}
    for row in records:
        key = (row["n_qubits"], row["state_type"], row["shots_per_setting"], row["method"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, rows in sorted(grouped.items()):
        summary.append(
            {
                "n_qubits": key[0],
                "state_type": key[1],
                "shots_per_setting": key[2],
                "method": key[3],
                "mean_fidelity": sum(r["fidelity"] for r in rows) / len(rows),
                "mean_hs_distance": sum(r["hs_distance"] for r in rows) / len(rows),
                "mean_reconstruction_seconds": sum(r["reconstruction_seconds"] for r in rows) / len(rows),
                "replicates": len(rows),
            }
        )
    return summary
