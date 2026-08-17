"""End-to-end and project-contract tests for the tomography pipeline.

AI disclosure: this test file was generated with OpenAI Codex assistance on
2026-08-17 and must not be marked verified until independently reviewed.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from nbqs_qst.measurement_generation import generate_measurement_dataset
from nbqs_qst.state_generation import (
    random_haar_state,
    random_product_state,
    random_state_with_purity,
)
from nbqs_qst.state_reconstruction import reconstruct, state_fidelity


FAMILIES = (
    (
        "product",
        lambda like: random_product_state(like, 2, seed=301),
    ),
    (
        "haar",
        lambda like: random_haar_state(like, 2, seed=302),
    ),
    (
        "mixed",
        lambda like: random_state_with_purity(
            like, 2, 0.55, seed=303
        ),
    ),
)


def test_core_modules_follow_static_hardware_agnostic_contracts():
    root = Path(__file__).resolve().parents[1]
    core_modules = (
        root / "src" / "nbqs_qst" / "pipeline.py",
        root / "src" / "nbqs_qst" / "state_generation" / "state_generation.py",
        root / "src" / "nbqs_qst" / "measurement_generation" / "pauli_measurement.py",
        root / "src" / "nbqs_qst" / "state_reconstruction" / "state_reconstruction.py",
    )
    forbidden_roots = {"numpy", "cupy", "jax", "torch"}

    for module in core_modules:
        source = module.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        assert not any(name.split(".")[0] in forbidden_roots for name in imports)
        backend_random_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "random"
            and not (isinstance(node.value, ast.Name) and node.value.id == "rng")
        ]
        assert not backend_random_calls
        assert module.with_suffix(".md").is_file()
        assert "AI disclosure" in source


@pytest.mark.parametrize(("family", "generate"), FAMILIES)
def test_numpy_full_pipeline_reconstructs_representative_families(family, generate):
    target = generate(np.asarray(0.0))
    dataset = generate_measurement_dataset(target.rho, shots=512, seed=401)
    linear = reconstruct(dataset, method="linear")
    pls = reconstruct(dataset, method="pls")
    mle = reconstruct(dataset, method="mle", initial="pls")

    assert dataset.settings == tuple(sorted(dataset.settings))
    assert len(dataset.settings) == 9
    assert linear.trace_error < 1e-12
    assert linear.hermiticity_error < 1e-12

    for physical in (pls, mle):
        assert physical.converged
        assert physical.trace_error < 1e-12
        assert physical.hermiticity_error < 1e-12
        assert physical.min_eigenvalue >= -1e-12
        assert state_fidelity(target.rho, physical.rho) > 0.95, family

    assert mle.objective <= pls.objective + 1e-12


@pytest.mark.parametrize(("family", "generate"), FAMILIES)
def test_numpy_and_jax_full_pipelines_match_for_fixed_seeds(family, generate):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")

    numpy_target = generate(np.asarray(0.0))
    jax_target = generate(jnp.asarray(0.0))
    numpy_data = generate_measurement_dataset(
        numpy_target.rho, shots=256, seed=402
    )
    jax_data = generate_measurement_dataset(
        jax_target.rho, shots=256, seed=402
    )

    assert np.allclose(numpy_target.rho, np.asarray(jax_target.rho), atol=2e-13)
    for numpy_counts, jax_counts in zip(numpy_data.counts, jax_data.counts):
        assert np.array_equal(numpy_counts, np.asarray(jax_counts))

    for method in ("linear", "pls", "mle"):
        kwargs = {"initial": "pls"} if method == "mle" else {}
        numpy_result = reconstruct(numpy_data, method=method, **kwargs)
        jax_result = reconstruct(jax_data, method=method, **kwargs)
        assert np.allclose(
            numpy_result.rho,
            np.asarray(jax_result.rho),
            atol=2e-10,
            rtol=2e-10,
        ), (family, method)
        assert numpy_result.objective == pytest.approx(
            jax_result.objective, abs=2e-12
        )
