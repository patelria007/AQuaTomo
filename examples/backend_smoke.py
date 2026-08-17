"""Run the same mathematical pipeline on NumPy and any installed accelerator."""

import importlib

import numpy as np

from nbqst.backend import backend_name, scalar
from nbqst.denoise import project_density_matrix
from nbqst.measurements import simulate_pauli_measurements
from nbqst.metrics import fidelity
from nbqst.reconstruction import linear_inversion_pauli
from nbqst.states import random_product_state


backends = [("numpy", np)]
for module, attribute in [("cupy", None), ("jax.numpy", None)]:
    try:
        backends.append((module, importlib.import_module(module)))
    except ImportError:
        pass

for label, xp in backends:
    truth = random_product_state(2, xp=xp, rng=5)
    data = simulate_pauli_measurements(truth, 500, rng=6)
    estimate = project_density_matrix(linear_inversion_pauli(data))
    print(label, backend_name(estimate), scalar(fidelity(truth, estimate)))

