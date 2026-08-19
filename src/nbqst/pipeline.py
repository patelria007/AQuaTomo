"""Object-oriented facade for the functional tomography kernels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .denoise import project_density_matrix
from .measurements import MeasurementData, simulate_pauli_measurements
from .metrics import fidelity, minimum_eigenvalue
from .neural import NeuralTomographyModel, neural_state_reconstruction
from .reconstruction import factorized_mle, linear_inversion_pauli
from .states import haar_random_pure, random_mixed_state, random_product_state


GENERATORS = {
    "product": random_product_state,
    "haar": haar_random_pure,
    "mixed": random_mixed_state,
}


@dataclass(frozen=True)
class ReconstructionResult:
    """One reconstructed state with metrics evaluated against known truth."""

    method: str
    estimate: object
    fidelity: float
    minimum_eigenvalue: float

    @property
    def is_physical(self) -> bool:
        return self.minimum_eigenvalue >= -1e-10


@dataclass
class TomographyPipeline:
    """Small reusable orchestration layer for examples and applications.

    Numerical arrays remain in ``xp``. The seeded NumPy generator is used only
    for reproducible state parameters and measurement sampling, operations not
    standardized by the Python Array API.
    """

    xp: object = np
    seed: int = 7
    mle_iterations: int = 100
    neural_models: dict[int, NeuralTomographyModel] = field(default_factory=dict)

    def __post_init__(self):
        if self.mle_iterations < 1:
            raise ValueError("mle_iterations must be positive")
        self.rng = np.random.default_rng(self.seed)

    def generate_state(self, n_qubits: int, *, state_type: str = "haar"):
        try:
            generator = GENERATORS[state_type]
        except KeyError as error:
            raise ValueError(f"unknown state_type {state_type!r}; choose from {tuple(GENERATORS)}") from error
        return generator(n_qubits, xp=self.xp, rng=self.rng)

    def measure(self, rho, shots_per_setting: int, **kwargs) -> MeasurementData:
        return simulate_pauli_measurements(rho, shots_per_setting, rng=self.rng, **kwargs)

    def reconstruct(self, data: MeasurementData, *, method: str):
        normalized = method.lower().replace("-", "_")
        if normalized in {"li", "linear", "linear_inversion"}:
            return linear_inversion_pauli(data)
        if normalized in {"projected_li", "physical_projection"}:
            return project_density_matrix(linear_inversion_pauli(data))
        if normalized in {"mle", "maximum_likelihood"}:
            return factorized_mle(data, max_iter=self.mle_iterations)
        if normalized in {"nn", "neural", "neural_network"}:
            try:
                model = self.neural_models[data.n_qubits]
            except KeyError as error:
                raise ValueError(f"no neural model is configured for {data.n_qubits} qubits") from error
            return neural_state_reconstruction(data, model)
        raise ValueError(f"unknown reconstruction method {method!r}")

    def evaluate(self, truth, estimate, *, method: str) -> ReconstructionResult:
        return ReconstructionResult(
            method=method,
            estimate=estimate,
            fidelity=float(fidelity(truth, estimate)),
            minimum_eigenvalue=float(minimum_eigenvalue(estimate)),
        )

    def run(
        self,
        *,
        n_qubits: int,
        shots_per_setting: int,
        state_type: str = "haar",
        methods=("li", "mle"),
        state_transform=None,
    ) -> tuple[object, MeasurementData, tuple[ReconstructionResult, ...]]:
        truth = self.generate_state(n_qubits, state_type=state_type)
        measured_state = truth if state_transform is None else state_transform(truth)
        data = self.measure(measured_state, shots_per_setting)
        results = tuple(
            self.evaluate(measured_state, self.reconstruct(data, method=method), method=method)
            for method in methods
        )
        return measured_state, data, results

