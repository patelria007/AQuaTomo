"""Minimal complete experiment: state -> shots -> reconstruction -> metrics."""

import numpy as np

from nbqst.backend import scalar
from nbqst.denoise import low_rank_projection, project_density_matrix
from nbqst.measurements import simulate_pauli_measurements
from nbqst.metrics import fidelity, minimum_eigenvalue
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli
from nbqst.states import haar_random_pure


rho_true = haar_random_pure(2, xp=np, rng=11)
data = simulate_pauli_measurements(rho_true, shots_per_setting=500, rng=12)
linear = linear_inversion_pauli(data)
projected = project_density_matrix(linear)
rank_one = low_rank_projection(linear, rank=1)
mle, history = factorized_mle(data, initial=projected, max_iter=100, return_history=True)

for name, estimate in {"linear": linear, "projected": projected, "rank_one": rank_one, "mle": mle}.items():
    print(
        f"{name:10s} fidelity={scalar(fidelity(rho_true, estimate)):.6f} "
        f"lambda_min={scalar(minimum_eigenvalue(estimate)):+.3e}"
    )
print(f"MLE iterations={len(history)-1}; NLL {history[0]:.8f} -> {history[-1]:.8f}")

