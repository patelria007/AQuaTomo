"""Build the small, self-explanatory NBQSS backend quick-start notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook():
    cells = [
        markdown(
            r"""
# NBQSS tomography: quick CPU/GPU backend check

This notebook is the shortest complete route through the submission. It uses
two qubits and modest shot counts so it should finish quickly on a laptop. The
same cells run on NumPy, CuPy, or JAX; only the backend selection changes.

The notebook checks state generation, physical noise, complete local-Pauli
measurements, linear inversion (LI), maximum-likelihood estimation (MLE), the
saved neural model, synchronized timers, and local-Pauli classical shadows.

> Reproducibility note: this submitted notebook was executed on the documented
> Apple M1 CPU with NumPy. CuPy and JAX are optional and are skipped when they
> are not installed. The experiment owner separately confirmed both accelerator
> commands on one NVIDIA RTX 4060 Ti (8 GB, sm 8.7); rerun the optional backend
> cell on that node to generate auditable timings and its manifest.
"""
        ),
        markdown(
            """
## 1. Locate the source tree

This makes the notebook work both from the repository root and from the
`notebooks/` directory. No package installation is needed for this quick check.
"""
        ),
        code(
            """
# Find the repository root and import the local source package.
from pathlib import Path
import sys

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
print("Repository:", ROOT.resolve())
"""
        ),
        markdown(
            """
## 2. Select an Array-API backend

Use `BACKEND = "numpy"` for any laptop. On a CUDA node choose `"cupy"` or
`"jax"` after installing the cluster-approved package. The helper reports
exactly which optional backends are importable instead of silently falling
back to the CPU.
"""
        ),
        code(
            """
# Import NumPy and expose optional accelerator namespaces without hiding failures.
import importlib.util
import numpy as np

def get_backend(name):
    if name == "numpy":
        return np
    if name == "cupy":
        import cupy as cp
        return cp
    if name == "jax":
        import jax.numpy as jnp
        return jnp
    raise ValueError(name)

available = {
    "numpy": True,
    "cupy": importlib.util.find_spec("cupy") is not None,
    "jax": importlib.util.find_spec("jax") is not None,
}
BACKEND = "numpy"  # Change only this line on the cluster.
xp = get_backend(BACKEND)
print("Available backends:", available)
print("Selected backend:", BACKEND)
"""
        ),
        markdown(
            """
## 3. Generate and verify a state

The object-oriented facade keeps the recipe compact while the numerical work
remains in modular Array-API functions. We explicitly check Hermiticity, unit
trace, positivity, and purity before using the generated state.
"""
        ),
        code(
            """
# Generate one Haar-random two-qubit pure state with a reproducible seed.
from nbqst.backend import to_numpy
from nbqst.pipeline import TomographyPipeline

pipeline = TomographyPipeline(xp=xp, seed=20260819, mle_iterations=8)
rho_ideal = pipeline.generate_state(2, state_type="haar")
rho_host = to_numpy(rho_ideal)
checks = {
    "Hermitian": np.allclose(rho_host, rho_host.conj().T),
    "trace one": np.isclose(np.trace(rho_host).real, 1.0),
    "positive semidefinite": np.linalg.eigvalsh(rho_host).min() >= -1e-10,
    "pure": np.isclose(np.trace(rho_host @ rho_host).real, 1.0),
}
print(checks)
assert all(checks.values())
"""
        ),
        markdown(
            """
## 4. Add a known noise channel

Global depolarization mixes the state with the maximally mixed state. The
trace and eigenvalue checks verify that the channel changed the state without
breaking the density-matrix constraints.
"""
        ),
        code(
            """
# Add 8% global depolarizing noise and verify the resulting density matrix.
from nbqst.metrics import fidelity
from nbqst.noise import global_depolarizing_channel

rho_noisy = global_depolarizing_channel(rho_ideal, 0.08)
noisy_host = to_numpy(rho_noisy)
print("Ideal-to-noisy fidelity:", float(fidelity(rho_ideal, rho_noisy)))
print("Noisy trace:", np.trace(noisy_host).real)
print("Noisy minimum eigenvalue:", np.linalg.eigvalsh(noisy_host).min())
assert np.isclose(np.trace(noisy_host).real, 1.0)
assert np.linalg.eigvalsh(noisy_host).min() >= -1e-10
"""
        ),
        markdown(
            """
## 5. Simulate complete local-Pauli data

For two qubits, informational completeness requires all `3**2 = 9` local
settings, not only `XX`, `YY`, and `ZZ`. Every setting below receives the same
number of multinomial shots.
"""
        ),
        code(
            """
# Measure every local X/Y/Z tensor-product setting with 250 shots per setting.
data = pipeline.measure(rho_noisy, shots_per_setting=250)
print("Settings:", data.settings)
print("Informationally complete:", data.informationally_complete)
print("Total measured copies:", len(data.settings) * data.shots_per_setting)
assert data.informationally_complete
assert all(int(to_numpy(counts).sum()) == 250 for counts in data.counts.values())
"""
        ),
        markdown(
            """
## 6. Reconstruct with LI and MLE

Raw LI is fast but can have negative eigenvalues at finite shots. MLE uses a
factorized density matrix and is physical by construction. Fidelity is
reported against the noisy state actually presented to the measurement.
"""
        ),
        code(
            """
# Reconstruct the same data with the two physics-based estimators.
from nbqst.metrics import minimum_eigenvalue

for method in ("li", "mle"):
    estimate = pipeline.reconstruct(data, method=method)
    print(
        method,
        "fidelity =", round(float(fidelity(rho_noisy, estimate)), 6),
        "minimum eigenvalue =", round(float(minimum_eigenvalue(estimate)), 8),
    )
"""
        ),
        markdown(
            """
## 7. Run the separate neural estimator

The bundled two-qubit network maps the same normalized Pauli frequencies to a
Cholesky factor, so its output is always physical. It is a separately trained
result, not part of the challenge specification. Training cost is not included
in inference.
"""
        ),
        code(
            """
# Load the saved two-qubit model and evaluate one inference call.
from nbqst.neural import load_neural_model, neural_state_reconstruction

model_path = ROOT / "results" / "neural_comparison_model.npz"
neural_model = load_neural_model(model_path)
neural_estimate = neural_state_reconstruction(data, neural_model)
print("Neural fidelity:", round(float(fidelity(rho_noisy, neural_estimate)), 6))
print("Neural minimum eigenvalue:", float(minimum_eigenvalue(neural_estimate)))
"""
        ),
        markdown(
            """
## 8. Estimate observables with classical shadows

One randomized local-Pauli shadow can be reused for several observables. The
example compares exact and estimated values for two one-body and two two-body
Pauli strings. Shadow error grows with observable weight because a weight-
`k` Pauli string matches only about one in `3**k` random settings.
"""
        ),
        code(
            """
# Acquire one reusable shadow and query four observables after measurement.
from nbqst.shadows import ClassicalShadowProtocol, observable_expectation

shadow_protocol = ClassicalShadowProtocol(median_of_means_groups=5)
shadow = shadow_protocol.acquire(rho_noisy, 3000, rng=20260820)
for label in ("ZI", "IZ", "ZZ", "XX"):
    result = shadow_protocol.estimate(shadow, label)
    exact = observable_expectation(rho_noisy, label)
    print(
        label,
        "exact =", round(exact, 4),
        "shadow =", round(result.value, 4),
        "empirical SE =", round(result.standard_error, 4),
    )
"""
        ),
        markdown(
            """
## 9. Run synchronized LI/MLE/NN timers

The benchmark separates reconstruction, fidelity evaluation, and complete
end-to-end time. CuPy streams and JAX arrays are synchronized before the clock
is read, so accelerator work cannot leak outside the timed interval.
"""
        ),
        code(
            """
# Time all three estimators on a tiny grid suitable for a quick correctness check.
from nbqst.experiment import benchmark

timings = benchmark(
    qubits=(2,),
    shots=(100, 500),
    state_types=("haar",),
    states_per_case=1,
    xp=xp,
    seed=20260821,
    methods=("li", "mle", "nn"),
    neural_models={2: neural_model},
    mle_iterations=4,
    warmup_rounds=1,
    timing_repeats=2,
)
for row in timings:
    print(
        row["backend"], row["method"], row["shots_per_setting"],
        "reconstruction =", f'{row["reconstruction_seconds"]:.6f}s',
        "fidelity =", f'{row["fidelity_seconds"]:.6f}s',
    )
"""
        ),
        markdown(
            """
## 10. Verify the same recipe on every installed backend

This cell is the backend portability check. It always runs NumPy and also runs
CuPy/JAX when present. A missing package is reported as a skip—not a successful
accelerator test. On the submitted CPU environment only NumPy is expected;
rerunning this cell on the cluster provides the second-backend demonstration.
"""
        ),
        code(
            """
# Execute one identical one-qubit pipeline per installed backend.
backend_results = {}
for name in ("numpy", "cupy", "jax"):
    if not available[name]:
        backend_results[name] = "SKIPPED: package unavailable"
        continue
    native_xp = get_backend(name)
    check_pipeline = TomographyPipeline(xp=native_xp, seed=99, mle_iterations=3)
    truth, _, results = check_pipeline.run(
        n_qubits=1,
        shots_per_setting=100,
        state_type="product",
        methods=("projected_li", "mle"),
    )
    backend_results[name] = {result.method: round(result.fidelity, 6) for result in results}
print(backend_results)
"""
        ),
        markdown(
            r"""
## 11. Cluster recipe and interpretation

Run these commands from the repository root on the allocated GPU node:

```bash
nbqst benchmark --backend cupy --qubits 1 2 3 4 5 --shots 100 1000 \
  --methods li mle --warmup-rounds 2 --timing-repeats 5

JAX_ENABLE_X64=1 nbqst benchmark --backend jax --qubits 1 2 3 4 5 \
  --shots 100 1000 --methods li mle --warmup-rounds 2 --timing-repeats 5
```

Preserve each CSV, summary CSV, manifest JSON, and scheduler log. A GPU can
accelerate dense kernels, but it cannot remove the `4**n` density matrix, the
`3**n` setting count, or the `6**n` table of setting/outcome frequencies.

To regenerate every analytical table and figure in the final submission:

```bash
python tools/run_extended_study.py --hardware-qubits 1 2 3 4 5
```
"""
        ),
        markdown(
            """
## 12. What to check before trusting a run

- LI fidelity is meaningful only when the estimate is physical; inspect its
  minimum eigenvalue.
- MLE and NN are constrained to physical states but can still be biased.
- A single targeted observable is usually best measured directly. Classical
  shadows become attractive when many observables must share one data set or
  are selected after acquisition.
- Compilation, host/device transfers, warm-up policy, precision, and GPU
  synchronization must be identical in CPU/GPU comparisons.
- Failure to reach 99% on a finite shot grid means “not reached,” not an
  extrapolated threshold.
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        }
    )
    return notebook


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("notebooks/NBQSS_Backend_Quickstart.ipynb"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
