"""Regression tests for the installed-style unified package API.

AI disclosure: this test file was generated with OpenAI Codex assistance on
2026-08-17 and must not be marked verified until independently reviewed.
"""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

import nbqs_qst as qst


def test_unified_api_runs_the_complete_pipeline():
    target = qst.random_haar_state(np.asarray(0.0), 2, seed=1701)
    run = qst.run_tomography(
        target,
        512,
        measurement_seed=1702,
        method="mle",
        initial="pls",
    )

    assert isinstance(run, qst.TomographyRun)
    assert len(run.measurements.settings) == 9
    assert run.reconstruction.converged
    assert run.physical_reconstruction
    assert run.fidelity is not None and run.fidelity > 0.95
    assert run.target_purity == pytest.approx(1.0, abs=1e-12)
    json.dumps(run.summary())


def test_linear_pipeline_does_not_report_fidelity_for_nonphysical_estimate():
    target = qst.random_product_state(np.asarray(0.0), 2, seed=701)
    run = qst.run_tomography(
        target,
        64,
        measurement_seed=8000,
        method="linear",
    )

    assert not run.physical_reconstruction
    assert run.reconstruction.min_eigenvalue < -1e-10
    assert run.fidelity is None


def test_pipeline_requires_generated_state():
    with pytest.raises(TypeError):
        qst.run_tomography(np.eye(2) / 2, 32)


def test_package_version_matches_project_metadata():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert qst.__version__ == metadata["project"]["version"]
