"""Regression tests for hardware-agnostic state reconstruction.

Run from the project root with::

    pytest -q state_reconstruction/reconstruction_test

AI disclosure: this test file was generated with AI assistance on 2026-08-17
and must not be marked verified until independently reviewed.
"""

from pathlib import Path

import numpy as np
import pytest

from nbqs_qst.measurement_generation import (
    MeasurementDataset,
    generate_measurement_dataset,
    pauli_expectations,
)
from nbqs_qst.state_reconstruction import state_reconstruction as sr


def density_matrix(state):
    state = np.asarray(state, dtype=np.complex128)
    return np.outer(state, state.conj())


def one_qubit_dataset(counts, shots=100, asarray=np.asarray):
    settings = ("X", "Y", "Z")
    count_arrays = tuple(
        asarray(values, dtype=np.int64) for values in counts
    )
    outcomes = tuple(asarray([], dtype=np.int64) for _ in settings)
    return MeasurementDataset(settings, outcomes, count_arrays, shots, 1, 0)


def test_linear_inversion_recovers_plus_y_and_imaginary_sign():
    dataset = one_qubit_dataset(([50, 50], [100, 0], [50, 50]))
    estimate = sr.linear_inversion(dataset)
    plus_y = np.asarray([1, 1j], dtype=np.complex128) / np.sqrt(2)
    expected = density_matrix(plus_y)

    assert np.allclose(estimate, expected, atol=1e-12)
    assert estimate[0, 1].imag == pytest.approx(-0.5)
    assert pauli_expectations(estimate)["Y"] == pytest.approx(1.0)


def test_linear_inversion_preserves_nonphysical_finite_data():
    dataset = one_qubit_dataset(([100, 0], [100, 0], [100, 0]))
    estimate = sr.linear_inversion(dataset)

    assert np.trace(estimate) == pytest.approx(1.0)
    assert np.allclose(estimate, estimate.conj().T)
    assert np.min(np.linalg.eigvalsh(estimate)) < -0.3


def test_projection_and_pls_are_physical():
    unphysical = np.diag([1.2, -0.2]).astype(np.complex128)
    projected = sr.project_density_matrix(unphysical)
    dataset = one_qubit_dataset(([100, 0], [100, 0], [100, 0]))
    pls = sr.projected_least_squares(dataset)

    assert np.allclose(projected, np.diag([1.0, 0.0]), atol=1e-12)
    for state in (projected, pls):
        assert np.trace(state) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(state, state.conj().T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(state)) >= -1e-12


def test_multinomial_mle_reaches_known_symmetric_solution():
    dataset = one_qubit_dataset(([100, 0], [100, 0], [100, 0]))
    mixed = np.eye(2, dtype=np.complex128) / 2
    result = sr.maximum_likelihood(dataset)

    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    expected = (np.eye(2) + (x + y + z) / np.sqrt(3)) / 2

    assert result.converged
    assert result.iterations > 0
    assert result.objective < sr.negative_log_likelihood(mixed, dataset)
    assert all(
        earlier >= later
        for earlier, later in zip(
            result.objective_history, result.objective_history[1:]
        )
    )
    assert result.trace_error < 1e-12
    assert result.hermiticity_error < 1e-12
    assert result.min_eigenvalue >= -1e-12
    assert sr.state_fidelity(expected, result.rho) > 1 - 1e-10


def test_two_qubit_bell_mle_is_accurate_and_physical():
    bell = np.asarray([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    target = density_matrix(bell)
    dataset = generate_measurement_dataset(target, shots=512, seed=12)
    result = sr.reconstruct(dataset, method="mle")

    assert result.converged
    assert result.min_eigenvalue >= -1e-12
    assert sr.state_fidelity(target, result.rho) > 0.99
    assert sr.purity(result.rho) <= 1 + 1e-12


def test_four_qubit_ghz_mle_is_accurate_and_physical():
    """Exercise all 81 Pauli settings of a nontrivial 16-dimensional state."""
    ghz = np.zeros(16, dtype=np.complex128)
    ghz[0] = 2**-0.5
    ghz[-1] = 2**-0.5
    target = density_matrix(ghz)
    dataset = generate_measurement_dataset(target, shots=256, seed=404)

    linear = sr.linear_inversion(dataset)
    result = sr.maximum_likelihood(
        dataset, max_iterations=500, tolerance=1e-8
    )

    assert len(dataset.settings) == 81
    assert np.min(np.linalg.eigvalsh(linear)) < -1e-3
    assert result.converged
    assert result.min_eigenvalue >= -1e-12
    assert result.trace_error < 1e-12
    assert result.hermiticity_error < 1e-12
    assert sr.state_fidelity(target, result.rho) > 0.995


def test_likelihood_gradient_matches_finite_difference():
    dataset = one_qubit_dataset(([61, 39], [47, 53], [58, 42]))
    rho = np.asarray(
        [[0.6, 0.1 + 0.05j], [0.1 - 0.05j, 0.4]],
        dtype=np.complex128,
    )
    direction = np.asarray(
        [[0.12, 0.04 - 0.03j], [0.04 + 0.03j, -0.12]],
        dtype=np.complex128,
    )
    _, _, _, _, total_shots = sr._validate_dataset(dataset)
    _, gradient = sr._cost_and_gradient(
        rho, dataset, np, total_shots, 1e-14
    )
    epsilon = 1e-6
    finite_difference = (
        sr.negative_log_likelihood(rho + epsilon * direction, dataset)
        - sr.negative_log_likelihood(rho - epsilon * direction, dataset)
    ) / (2 * epsilon)
    analytic = float(np.real(np.sum(gradient.conj() * direction)))

    assert analytic == pytest.approx(finite_difference, rel=2e-6, abs=2e-8)


def test_linear_error_scales_as_inverse_sqrt_shots():
    target = np.asarray(
        [[0.7, 0.15 + 0.1j], [0.15 - 0.1j, 0.3]],
        dtype=np.complex128,
    )
    shot_counts = (64, 256, 1024, 4096)
    trials = 16
    rms_errors = []

    for shots in shot_counts:
        squared_errors = []
        for trial in range(trials):
            dataset = generate_measurement_dataset(
                target, shots=shots, seed=1000 + trial
            )
            estimate = sr.linear_inversion(dataset)
            squared_errors.append(np.linalg.norm(estimate - target) ** 2)
        rms_errors.append(float(np.sqrt(np.mean(squared_errors))))

    slope = np.polyfit(np.log(shot_counts), np.log(rms_errors), 1)[0]
    assert -0.65 < slope < -0.35
    assert rms_errors[-1] < rms_errors[0] / 5


def test_metrics_follow_project_definitions():
    plus = density_matrix(np.asarray([1, 1]) / np.sqrt(2))
    zero = density_matrix([1, 0])
    mixed = np.eye(2, dtype=np.complex128) / 2

    assert sr.purity(plus) == pytest.approx(1.0)
    assert sr.purity(mixed) == pytest.approx(0.5)
    assert sr.state_fidelity(plus, plus) == pytest.approx(1.0)
    assert sr.state_fidelity(zero, plus) == pytest.approx(0.5)
    assert sr.trace_distance(zero, plus) == pytest.approx(1 / np.sqrt(2))


def test_fixed_counts_match_between_numpy_and_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")
    counts = ([83, 17], [29, 71], [64, 36])
    numpy_dataset = one_qubit_dataset(counts)
    jax_dataset = one_qubit_dataset(counts, asarray=jnp.asarray)

    numpy_linear = sr.linear_inversion(numpy_dataset)
    jax_linear = sr.linear_inversion(jax_dataset)
    numpy_mle = sr.maximum_likelihood(numpy_dataset)
    jax_mle = sr.maximum_likelihood(jax_dataset)

    assert np.allclose(numpy_linear, np.asarray(jax_linear), atol=1e-12)
    assert np.allclose(numpy_mle.rho, np.asarray(jax_mle.rho), atol=1e-10)
    assert numpy_mle.objective == pytest.approx(jax_mle.objective, abs=1e-12)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: sr.linear_inversion(object()), TypeError),
        (
            lambda: sr.linear_inversion(
                one_qubit_dataset(([50, 50], [50, 50], [49, 50]))
            ),
            ValueError,
        ),
        (
            lambda: sr.maximum_likelihood(
                one_qubit_dataset(([50, 50], [50, 50], [50, 50])),
                max_iterations=0,
            ),
            ValueError,
        ),
        (
            lambda: sr.reconstruct(
                one_qubit_dataset(([50, 50], [50, 50], [50, 50])),
                method="unknown",
            ),
            ValueError,
        ),
    ],
)
def test_invalid_inputs_fail_clearly(call, error):
    with pytest.raises(error):
        call()
