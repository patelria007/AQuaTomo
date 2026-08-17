# State-generation poster plots

> **AI disclosure:** The plotting code, figure composition, captions, and this
> text were generated with OpenAI Codex assistance on 2026-08-17. The figures
> remain unverified until independently reviewed.

Run from the repository root:

```bash
python state_generation/state_test/generate_poster_plots.py
```

The script writes both 300-dpi PNG and vector PDF versions of four figures.
The figures intentionally contain only figure titles, axis/subplot titles,
ticks, legends, and color scales. Explanations, formulas, sample details, and
validation status are collected in
[`state_generation_report.md`](../../reports/state_generation_report.md).

## `state_families_1_to_6_qubits`

- Columns cover every size from 1Q (`2 x 2`) through 6Q (`64 x 64`).
- Rows cover local-Haar product pure, global Haar pure, and induced mixed
  (`K=d`) states at one common fixed seed (`29`).
- Each density-matrix magnitude is normalized by its own largest entry so the
  internal pattern remains visible as dimension grows. Color intensity must
  therefore not be compared as an absolute matrix-element value across panels.
- At 6Q the state visualization is still small, while exhaustive Pauli
  measurement requires `3**6 = 729` settings; these are distinct costs.

## `three_generation_families`

- The three columns show the challenge's required generators together:
  local-Haar product pure, global Haar pure, and induced mixed with `K=d`.
- The top row compares density-matrix magnitudes `|rho_ij|` on one shared color
  scale. The bottom row gives the full eigenvalue spectrum.
- Numerical rank, purity, global von Neumann entropy, and reduced-state entropy
  are reported in the companion report rather than inside the figure.

## `pure_state_ensembles`

- Panel A shows 320 independently seeded one-qubit Haar states as Bloch vectors.
  Product-state local factors use the same distribution.
- Panel B compares 2|2 bipartite von Neumann entropy for 240 four-qubit product
  states and 240 four-qubit global Haar states. Both families are pure, so the
  contrast isolates entanglement structure rather than purity.
- The dashed reference shows Page's Haar-average entropy for a 4 x 4
  bipartition.

## `mixed_state_purity`

- Panel A compares 120 generated states per `K` at `d=8` with the analytical
  induced-ensemble prediction. Error bars are 95% confidence intervals for the
  state-to-state sample mean.
- Panel B checks that the depolarized Haar and product families attain their
  requested analytical purity across the full physical interval `[1/d, 1]`.

No backend random-number generator is used. Every plotted state comes from the
public generator with a fixed stdlib seed. NumPy is used only for offline
analysis and Matplotlib rendering, not in the hardware-agnostic core module.

The PDF files are preferred for poster layout because text and lines remain
vector graphics. PNG files are convenient for previews and raster workflows.
