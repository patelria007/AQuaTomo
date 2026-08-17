# Full-stack validation figure generator

This companion note documents `generate_validation_figures.py`. The script
runs the complete state-generation → finite-shot Pauli measurement → state
reconstruction pipeline for two-qubit product, Haar-random pure, and
purity-controlled mixed states.

For each state family it uses 64, 256, and 1,024 shots per setting and 12 fixed
seeds. Linear inversion is assessed for physicality. Fidelity is reported only
for the physical PLS and MLE estimates because squared Uhlmann fidelity is a
density-matrix metric. The script also runs identical NumPy and JAX pipelines
and records the maximum absolute difference at each stage.

Run from the project root:

```powershell
python tests/generate_validation_figures.py
```

The script writes machine-readable JSON plus PNG and PDF figures to
`docs/validation_artifacts/`. Randomness enters only through the
project APIs, which use `random.Random`; NumPy is used only to aggregate results
and Matplotlib is used only for rendering.

AI disclosure: this script, experiment design, figure design, and companion
note were generated with OpenAI Codex assistance on 2026-08-17. The fixed-seed
outputs have been exercised by automated tests, but independent scientific and
software review remains pending.
