"""High-level connection between generation, measurement, and reconstruction.

The state generator remains explicit: callers create a ``GeneratedState`` and
pass it to :func:`run_tomography`.  The function then performs one complete
finite-shot Pauli experiment, reconstructs the state, and returns immutable
diagnostics in a single record.

AI disclosure: this module was generated with OpenAI Codex assistance on
2026-08-17. It must not be marked verified until independently reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .measurement_generation import MeasurementDataset, generate_measurement_dataset
from .state_generation import GeneratedState, state_purity
from .state_reconstruction import (
    ReconstructionResult,
    purity,
    reconstruct,
    state_fidelity,
)

__all__ = ["TomographyRun", "run_tomography"]


@dataclass(frozen=True)
class TomographyRun:
    """Immutable outputs and quality metrics from one complete QST run."""

    target: GeneratedState
    measurements: MeasurementDataset
    reconstruction: ReconstructionResult
    target_purity: float
    reconstructed_purity: float
    fidelity: float | None

    @property
    def physical_reconstruction(self):
        """Whether standard trace, Hermiticity, and PSD tolerances pass."""
        result = self.reconstruction
        return (
            result.trace_error <= 1e-10
            and result.hermiticity_error <= 1e-10
            and result.min_eigenvalue >= -1e-10
        )

    def summary(self):
        """Return a JSON-serializable summary without transferring matrices."""
        result = self.reconstruction
        return {
            "family": self.target.family,
            "num_qubits": self.target.num_qubits,
            "shots_per_setting": self.measurements.shots_per_setting,
            "measurement_seed": self.measurements.seed,
            "method": result.method,
            "converged": result.converged,
            "iterations": result.iterations,
            "objective": result.objective,
            "physical_reconstruction": self.physical_reconstruction,
            "target_purity": self.target_purity,
            "reconstructed_purity": self.reconstructed_purity,
            "fidelity": self.fidelity,
            "trace_error": result.trace_error,
            "hermiticity_error": result.hermiticity_error,
            "minimum_eigenvalue": result.min_eigenvalue,
        }


def run_tomography(
    target,
    shots,
    *,
    measurement_seed=None,
    method="mle",
    **reconstruction_options,
):
    """Run measurement and reconstruction for a generated target state.

    ``fidelity`` is returned only when the reconstruction is physical within
    the package tolerance.  Linear inversion can be non-positive after finite
    sampling, and squared Uhlmann fidelity is not defined for such a matrix.
    """
    if not isinstance(target, GeneratedState):
        raise TypeError("target must be a GeneratedState")

    measurements = generate_measurement_dataset(
        target.rho,
        shots,
        seed=measurement_seed,
    )
    result = reconstruct(
        measurements,
        method=method,
        **reconstruction_options,
    )
    target_purity = state_purity(target.rho)
    reconstructed_purity = purity(result.rho)
    physical = (
        result.trace_error <= 1e-10
        and result.hermiticity_error <= 1e-10
        and result.min_eigenvalue >= -1e-10
    )
    fidelity = state_fidelity(target.rho, result.rho) if physical else None
    return TomographyRun(
        target=target,
        measurements=measurements,
        reconstruction=result,
        target_purity=target_purity,
        reconstructed_purity=reconstructed_purity,
        fidelity=fidelity,
    )
