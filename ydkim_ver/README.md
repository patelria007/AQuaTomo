# NBQSS Quantum State Tomography Package

This directory is the standalone, shareable distribution of the
hardware-agnostic small-system QST project. It does not depend on files outside
this folder.

## Layout

```text
package/
├── pyproject.toml
├── src/
│   └── nbqs_qst/
│       ├── state_generation/
│       ├── measurement_generation/
│       ├── state_reconstruction/
│       └── pipeline.py
├── examples/
│   └── complete_qst_pipeline.ipynb
├── tests/
├── docs/
│   ├── reports/
│   ├── figures/
│   └── validation_artifacts/
└── dist/
```

## Install

From this directory:

```powershell
python -m pip install -e ".[example,test,dev]"
```

JAX users must enable 64-bit mode before creating arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Complete pipeline

```python
import numpy as np
import nbqs_qst as qst

target = qst.random_haar_state(np.asarray(0.0), 2, seed=17)
run = qst.run_tomography(
    target,
    shots=512,
    measurement_seed=23,
    method="mle",
    initial="pls",
)
print(run.summary())
```

The measurement simulator performs i.i.d. single-shot draws from exact Born
probabilities in all X/Y/Z settings. It does not add Gaussian noise. Randomness
comes only from stdlib `random.Random`.

## Test, notebook, and build

```powershell
python -m pytest -q
python -m nbconvert --to notebook --execute --inplace examples/complete_qst_pipeline.ipynb
python -m build
```

The executed notebook is
[`examples/complete_qst_pipeline.ipynb`](examples/complete_qst_pipeline.ipynb).
The full audit is
[`docs/full_stack_validation_report.md`](docs/full_stack_validation_report.md).

## Scope

This is exhaustive dense tomography for small systems. NumPy and JAX CPU are
tested. GPU, CuPy, PyTorch, and 20-qubit full tomography require separate
validation or a reduced protocol such as classical shadows.

## AI disclosure

Code, tests, derivations, documentation, figures, and the example notebook were
created or revised with generative-AI assistance. Automated validation is
included, but independent human scientific and software review is required
before any result is labelled verified.
