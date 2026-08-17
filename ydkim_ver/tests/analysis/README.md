# Reproducible analysis scripts

These scripts regenerate the component validation figures under
`docs/figures/`:

```powershell
python tests/analysis/generate_state_generation_figures.py
python tests/analysis/generate_measurement_figures.py
python tests/analysis/generate_reconstruction_figures.py
```

The full-stack experiment is generated separately with
`python tests/generate_validation_figures.py`.

Analysis scripts may use NumPy and Matplotlib; numerical package logic remains
Array-API portable. All quantum states and finite-shot data are created through
the public package API and its stdlib RNG path.

AI disclosure: these scripts, figure designs, and this guide were generated or
revised with OpenAI Codex assistance. Independent review is pending.
