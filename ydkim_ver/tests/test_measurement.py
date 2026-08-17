"""Fast regression tests for :mod:`measurement_generation.pauli_measurement`.

These tests intentionally contain no plotting or performance benchmarks.
Poster figures and scaling benchmarks belong in a separate, manually run
script so the normal test loop stays quick.

Run with::

    pytest -q measurement_generation/measure_test

AI disclosure: this test file was generated with AI assistance on 2026-08-17
and must not be marked verified until independently reviewed.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

from nbqs_qst.measurement_generation import pauli_measurement as pm


def state_vector(psi):
    """Return the density matrix of a pure state vector."""
    return np.outer(psi, psi.conj())


def test_exact_reference_states_and_plus_y_sign():
    """Check Born-rule values, including the odd-Y transpose regression."""
    plus_y = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    bell = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)

    plus_y_result = pm.pauli_expectations(state_vector(plus_y))
    bell_result = pm.pauli_expectations(state_vector(bell))

    assert plus_y_result == pytest.approx({"I": 1, "X": 0, "Y": 1, "Z": 0}, abs=1e-12)
    bell_expected = {"II": 1, "XX": 1, "YY": -1, "ZZ": 1, "XI": 0, "ZI": 0}
    for pauli, expected in bell_expected.items():
        assert bell_result[pauli] == pytest.approx(expected, abs=1e-12)


def test_all_two_qubit_paulis_match_direct_born_rule():
    """Cross-check the vectorized trace convention on an asymmetric state."""
    ket = np.asarray([1, 2j, -0.5 + 0.3j, 0.7 - 1.1j], dtype=np.complex128)
    ket = ket / np.linalg.norm(ket)
    rho = state_vector(ket)
    labels = pm.pauli_strings(2)
    matrices = np.asarray(pm.pauli_matrices(2, np))
    actual = pm.pauli_expectations(rho)

    for label, pauli in zip(labels, matrices):
        expected = np.trace(pauli @ rho).real
        assert actual[label] == pytest.approx(expected, abs=2e-13)


def test_seed_is_reproducible():
    plus_x = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    rho = state_vector(plus_x)

    first = pm.pauli_expectations(rho, shots=256, seed=17)
    second = pm.pauli_expectations(rho, shots=256, seed=17)

    assert first == second


def test_sampling_error_scales_as_inverse_sqrt_shots():
    """Small deterministic statistical check, not a poster-quality study."""
    plus_x = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    rho = state_vector(plus_x)
    shot_counts = (64, 256, 1024, 4096)
    trials = 12
    rms_errors = []

    for shots in shot_counts:
        errors = []
        for seed in range(trials):
            estimate = pm.pauli_expectations(rho, shots=shots, seed=seed)["Z"]
            errors.append(estimate)  # exact <Z> is zero
        rms_errors.append(float(np.sqrt(np.mean(np.square(errors)))))

    slope = np.polyfit(np.log(shot_counts), np.log(rms_errors), 1)[0]
    assert -0.7 < slope < -0.3
    assert rms_errors[-1] < rms_errors[0]


def test_dataset_retains_counts_and_aggregates_all_compatible_settings():
    """Lower-support Paulis must use every compatible setting, not the last."""
    plus_y = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    zero = np.array([1, 0], dtype=np.complex128)
    psi = np.kron(plus_y, zero)
    rho = state_vector(psi)
    shots = 73

    dataset = pm.generate_measurement_dataset(rho, shots, seed=41)
    estimates = pm.expectations_from_dataset(dataset)

    assert len(dataset.settings) == 9
    for outcomes, counts in zip(dataset.outcomes, dataset.counts):
        outcomes_np = np.asarray(outcomes)
        counts_np = np.asarray(counts)
        assert outcomes_np.shape == (shots,)
        assert counts_np.sum() == shots
        assert np.array_equal(counts_np, np.bincount(outcomes_np, minlength=4))
        assert np.all((0 <= outcomes_np) & (outcomes_np < 4))

    # XI is compatible with XX, XY, and XZ. Recompute its estimate from all
    # three raw arrays and require the public aggregation to match.
    xi_parity_sum = 0
    compatible_settings = 0
    for basis, outcomes in zip(dataset.settings, dataset.outcomes):
        if basis[0] == "X":
            outcomes_np = np.asarray(outcomes)
            xi_parity_sum += np.sum(1 - 2 * ((outcomes_np >> 1) & 1))
            compatible_settings += 1
    assert compatible_settings == 3
    assert estimates["XI"] == pytest.approx(
        xi_parity_sum / (compatible_settings * shots)
    )
    assert pm.pauli_expectations(rho, shots=shots, seed=41) == estimates


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: pm.pauli_strings(True), TypeError),
        (lambda: pm.pauli_expectations(np.eye(3) / 3), ValueError),
        (lambda: pm.pauli_expectations(np.eye(2)), ValueError),
        (
            lambda: pm.pauli_expectations(
                np.asarray([[0.5, 1.0], [0.0, 0.5]], dtype=np.complex128)
            ),
            ValueError,
        ),
        (
            lambda: pm.sample_outcomes(
                np.diag([1.2, -0.2]).astype(np.complex128), "Z", 8
            ),
            ValueError,
        ),
        (
            lambda: pm.generate_measurement_dataset(
                np.asarray([[np.nan, 0.0], [0.0, np.nan]], dtype=np.complex128),
                8,
            ),
            ValueError,
        ),
        (lambda: pm.pauli_expectations(np.eye(2) / 2, shots=0), ValueError),
        (lambda: pm.sample_outcomes(np.eye(2) / 2, "A", 8), ValueError),
        (lambda: pm.sample_outcomes(np.eye(2) / 2, "ZZ", 8), ValueError),
        (lambda: pm.sample_outcomes(np.eye(2) / 2, "Z", 1.5), TypeError),
    ],
)
def test_invalid_inputs_fail_clearly(call, error):
    with pytest.raises(error):
        call()


def test_most_significant_qubit_outcome_order_and_pauli_lexicography():
    """Use asymmetric |01> data to disambiguate both ordering contracts."""
    ket = np.asarray([0, 1, 0, 0], dtype=np.complex128)
    rho = state_vector(ket)

    outcomes = pm.sample_outcomes(rho, "ZZ", 16, seed=5)
    expectations = pm.pauli_expectations(rho)
    labels = pm.pauli_strings(2)
    matrices = np.asarray(pm.pauli_matrices(2, np))

    assert np.array_equal(outcomes, np.ones(16, dtype=np.int64))
    assert expectations["ZI"] == pytest.approx(1.0)
    assert expectations["IZ"] == pytest.approx(-1.0)
    assert labels == sorted(labels)
    assert labels[:6] == ["II", "IX", "IY", "IZ", "XI", "XX"]
    assert np.array_equal(matrices[labels.index("XZ")], np.kron(
        np.asarray([[0, 1], [1, 0]]),
        np.asarray([[1, 0], [0, -1]]),
    ))


def _installed_backends():
    """Yield installed array backends without making any backend RNG calls."""
    yield "numpy", np.asarray

    try:
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = importlib.import_module("jax.numpy")
        yield "jax", jnp.asarray
    except ImportError:
        pass

    for name in ("cupy", "torch"):
        try:
            module = importlib.import_module(name)
            yield name, module.asarray
        except ImportError:
            pass


def test_fixed_seed_matches_across_installed_backends():
    """A fixed stdlib seed must produce the same result on every backend."""
    plus_y = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    rho_np = state_vector(plus_y)
    expected = pm.pauli_expectations(rho_np, shots=256, seed=23)
    expected_dataset = pm.generate_measurement_dataset(rho_np, shots=64, seed=29)

    checked = []
    for name, asarray in _installed_backends():
        try:
            rho = asarray(rho_np)
            result = pm.pauli_expectations(rho, shots=256, seed=23)
            dataset = pm.generate_measurement_dataset(rho, shots=64, seed=29)
        except RuntimeError as exc:
            if name == "cupy":
                continue
            raise
        assert result == expected, f"fixed-seed result differs on {name}"
        assert dataset.settings == expected_dataset.settings
        for actual, reference in zip(dataset.outcomes, expected_dataset.outcomes):
            assert np.array_equal(np.asarray(actual), np.asarray(reference))
        for actual, reference in zip(dataset.counts, expected_dataset.counts):
            assert np.array_equal(np.asarray(actual), np.asarray(reference))
        checked.append(name)

    assert "numpy" in checked
    # This project requires a demonstrated multi-backend pipeline rather than
    # allowing the compatibility test to pass silently on NumPy alone.
    assert len(checked) >= 2, "install JAX, CuPy, or PyTorch to test a second backend"
