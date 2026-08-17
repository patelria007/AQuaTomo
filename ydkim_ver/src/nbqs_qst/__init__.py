"""Unified public API for the NBQSS hardware-agnostic QST project.

AI disclosure: this package API was generated with OpenAI Codex assistance on
2026-08-17. It must not be marked verified until independently reviewed.
"""

from .measurement_generation import (
    MeasurementDataset,
    expectations_from_dataset,
    generate_measurement_dataset,
    pauli_expectations,
    pauli_matrices,
    pauli_strings,
    sample_outcomes,
)
from .state_generation import (
    GeneratedState,
    ghz_state,
    pure_state_overlap,
    quantum_state_fidelity,
    random_haar_state,
    random_mixed_state,
    random_product_state,
    random_state_with_purity,
    state_purity,
    w_state,
)
from .state_reconstruction import (
    ReconstructionResult,
    linear_inversion,
    maximum_likelihood,
    negative_log_likelihood,
    project_density_matrix,
    projected_least_squares,
    purity,
    reconstruct,
    state_fidelity,
    trace_distance,
)

from .pipeline import TomographyRun, run_tomography

__version__ = "0.1.0"

__all__ = [
    "GeneratedState",
    "MeasurementDataset",
    "ReconstructionResult",
    "TomographyRun",
    "expectations_from_dataset",
    "generate_measurement_dataset",
    "ghz_state",
    "linear_inversion",
    "maximum_likelihood",
    "negative_log_likelihood",
    "pauli_expectations",
    "pauli_matrices",
    "pauli_strings",
    "project_density_matrix",
    "projected_least_squares",
    "pure_state_overlap",
    "purity",
    "quantum_state_fidelity",
    "random_haar_state",
    "random_mixed_state",
    "random_product_state",
    "random_state_with_purity",
    "reconstruct",
    "run_tomography",
    "sample_outcomes",
    "state_fidelity",
    "state_purity",
    "trace_distance",
    "w_state",
]
