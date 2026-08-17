"""Regression tests for :mod:`state_generation.state_generation`.

AI disclosure: this test file was generated with OpenAI Codex assistance on
2026-08-17 and must not be marked verified until independently reviewed.
"""

import ast
import importlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from nbqs_qst.measurement_generation import generate_measurement_dataset
from nbqs_qst.state_generation import (
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


LIKE = np.asarray(0.0)


def _assert_physical(state, expected_purity=None):
    rho = np.asarray(state.rho)
    d = 2**state.num_qubits
    assert rho.shape == (d, d)
    assert rho.dtype == np.complex128
    assert np.allclose(rho, rho.conj().T, atol=1e-13)
    assert np.trace(rho) == pytest.approx(1.0, abs=1e-13)
    assert np.linalg.eigvalsh(rho).min() >= -1e-12
    if expected_purity is not None:
        assert state_purity(state.rho) == pytest.approx(expected_purity, abs=2e-12)


def test_all_families_are_physical_and_metadata_is_reproducible():
    states = (
        random_product_state(LIKE, 3, seed=1),
        random_haar_state(LIKE, 3, seed=2),
        random_mixed_state(LIKE, 3, k=5, seed=3),
        random_state_with_purity(LIKE, 3, 0.4, seed=4),
        ghz_state(LIKE, 3),
        w_state(LIKE, 3),
    )
    for state in states:
        _assert_physical(state)
        metadata = state.metadata()
        assert metadata["family"] == state.family
        assert metadata["dimension"] == 8

    with pytest.raises(FrozenInstanceError):
        states[0].family = "changed"


def test_pure_families_have_unit_purity_and_normalized_kets():
    for state in (
        random_product_state(LIKE, 4, seed=11),
        random_haar_state(LIKE, 4, seed=12),
        ghz_state(LIKE, 4),
        w_state(LIKE, 4),
    ):
        _assert_physical(state, expected_purity=1.0)
        assert np.vdot(np.asarray(state.ket), np.asarray(state.ket)).real == pytest.approx(
            1.0, abs=1e-13
        )


def test_product_state_is_unentangled_across_a_bipartition():
    state = random_product_state(LIKE, 4, seed=17)
    coefficient_matrix = np.asarray(state.ket).reshape(4, 4)
    schmidt_values = np.linalg.svd(coefficient_matrix, compute_uv=False)
    assert schmidt_values[0] == pytest.approx(1.0, abs=1e-13)
    assert np.linalg.norm(schmidt_values[1:]) < 1e-13


def test_six_qubit_required_families_are_physical():
    product = random_product_state(LIKE, 6, seed=19)
    haar = random_haar_state(LIKE, 6, seed=20)
    mixed = random_mixed_state(LIKE, 6, k=64, seed=21)

    _assert_physical(product, expected_purity=1.0)
    _assert_physical(haar, expected_purity=1.0)
    _assert_physical(mixed)
    assert product.rho.shape == haar.rho.shape == mixed.rho.shape == (64, 64)
    assert np.linalg.matrix_rank(np.asarray(mixed.rho), tol=1e-11) == 64


def test_ghz_and_w_amplitudes_follow_most_significant_qubit_order():
    ghz = np.asarray(ghz_state(LIKE, 3).ket)
    w = np.asarray(w_state(LIKE, 3).ket)
    scale_ghz = 1 / np.sqrt(2)
    scale_w = 1 / np.sqrt(3)

    expected_ghz = np.asarray(
        [scale_ghz if index in (0, 7) else 0 for index in range(8)]
    )
    expected_w = np.asarray(
        [scale_w if index in (4, 2, 1) else 0 for index in range(8)]
    )
    assert np.allclose(ghz, expected_ghz)
    assert np.allclose(w, expected_w)


def test_seed_is_reproducible_and_distinguishes_samples():
    first = random_haar_state(LIKE, 3, seed=29)
    second = random_haar_state(LIKE, 3, seed=29)
    different = random_haar_state(LIKE, 3, seed=30)
    assert np.array_equal(first.ket, second.ket)
    assert np.array_equal(first.rho, second.rho)
    assert not np.array_equal(first.ket, different.ket)


def test_induced_state_rank_and_mean_purity_formula():
    d = 4
    k = 6
    samples = [random_mixed_state(LIKE, 2, k=k, seed=seed) for seed in range(128)]
    for state in samples[:8]:
        _assert_physical(state)
        assert np.linalg.matrix_rank(np.asarray(state.rho), tol=1e-11) == d

    observed = np.mean([state.purity for state in samples])
    expected = (d + k) / (d * k + 1)
    assert observed == pytest.approx(expected, abs=0.018)

    rank_two = random_mixed_state(LIKE, 3, k=2, seed=31)
    assert np.linalg.matrix_rank(np.asarray(rank_two.rho), tol=1e-11) == 2

    rank_one = random_mixed_state(LIKE, 3, k=1, seed=32)
    _assert_physical(rank_one, expected_purity=1.0)


@pytest.mark.parametrize("target", [1 / 8, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("base", ["haar", "product"])
def test_target_purity_is_exact(target, base):
    state = random_state_with_purity(LIKE, 3, target, seed=37, base=base)
    _assert_physical(state, expected_purity=target)
    assert (state.ket is not None) is (target == 1.0)
    assert state.metadata()["parameters"]["target_purity"] == target


def test_metrics_follow_challenge_conventions():
    ghz = ghz_state(LIKE, 3)
    w = w_state(LIKE, 3)
    maximally_mixed = np.eye(8, dtype=np.complex128) / 8
    phased_ghz = np.exp(0.37j) * np.asarray(ghz.ket)

    assert pure_state_overlap(ghz.ket, phased_ghz) == pytest.approx(1.0, abs=1e-13)
    assert pure_state_overlap(ghz.ket, w.ket) == pytest.approx(0.0, abs=1e-13)
    assert quantum_state_fidelity(ghz.rho, ghz.rho) == pytest.approx(1.0, abs=1e-13)
    assert quantum_state_fidelity(ghz.rho, w.rho) == pytest.approx(0.0, abs=1e-13)
    assert quantum_state_fidelity(ghz.rho, maximally_mixed) == pytest.approx(
        1 / 8, abs=1e-12
    )


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: random_haar_state(LIKE, 0), ValueError),
        (lambda: random_product_state(LIKE, 1.5), TypeError),
        (lambda: random_mixed_state(LIKE, 2, k=0), ValueError),
        (lambda: random_mixed_state(LIKE, 2, k=True), TypeError),
        (lambda: random_state_with_purity(LIKE, 2, 0.1), ValueError),
        (lambda: random_state_with_purity(LIKE, 2, 1.1), ValueError),
        (lambda: random_state_with_purity(LIKE, 2, 0.5, base="unknown"), ValueError),
        (lambda: state_purity(np.ones(3)), ValueError),
        (
            lambda: pure_state_overlap(np.ones(2), np.ones(4)),
            ValueError,
        ),
        (
            lambda: quantum_state_fidelity(np.eye(2) / 2, np.eye(4) / 4),
            ValueError,
        ),
    ],
)
def test_invalid_inputs_fail_clearly(call, error):
    with pytest.raises(error):
        call()


def _installed_backends():
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


def test_fixed_seed_matches_across_installed_backends_and_measurement_pipeline():
    factories = (
        lambda like: random_product_state(like, 2, seed=41),
        lambda like: random_haar_state(like, 2, seed=42),
        lambda like: random_mixed_state(like, 2, k=3, seed=43),
        lambda like: random_state_with_purity(like, 2, 0.6, seed=44),
    )
    references = [factory(LIKE) for factory in factories]
    reference_dataset = generate_measurement_dataset(
        references[0].rho, shots=48, seed=45
    )

    checked = []
    for name, asarray in _installed_backends():
        try:
            like = asarray(0.0)
            states = [factory(like) for factory in factories]
            dataset = generate_measurement_dataset(states[0].rho, shots=48, seed=45)
        except RuntimeError:
            if name == "cupy":
                continue
            raise

        for actual, reference in zip(states, references):
            assert np.asarray(actual.rho).dtype == np.complex128
            assert np.allclose(
                np.asarray(actual.rho), np.asarray(reference.rho), atol=2e-13, rtol=2e-13
            )
            if reference.ket is not None:
                assert np.allclose(
                    np.asarray(actual.ket),
                    np.asarray(reference.ket),
                    atol=2e-13,
                    rtol=2e-13,
                )

        for actual, reference in zip(dataset.outcomes, reference_dataset.outcomes):
            assert np.array_equal(np.asarray(actual), np.asarray(reference))
        checked.append(name)

    assert "numpy" in checked
    assert len(checked) >= 2, "install JAX, CuPy, or PyTorch for a second backend"


def test_core_module_has_no_numpy_import_or_backend_rng():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "nbqs_qst"
        / "state_generation"
        / "state_generation.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert not any(name == "numpy" or name.startswith("numpy.") for name in imports)
    assert ".random" not in source
