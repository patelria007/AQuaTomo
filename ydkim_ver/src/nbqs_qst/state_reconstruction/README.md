# State reconstruction

This package reconstructs density matrices from the backend-native counts in
`measurement_generation.MeasurementDataset`.

- `state_reconstruction.py`: Pauli linear inversion, projected least squares,
  exact multinomial maximum-likelihood estimation, fidelity, purity, and trace
  distance. This is the package's single reconstruction implementation file.
- `state_reconstruction.md`: per-function theory and API companion.
- `../../../docs/reports/state_reconstruction_report.md`: English report with
  final figures, numerical result tables, limitations, and instructions.
- `../../../tests/test_reconstruction.py`: regression tests.
- `../../../tests/analysis/generate_reconstruction_figures.py`: figures.

Research and implementation design:

- [`theory_notes.md`](theory_notes.md): Pauli linear inversion, projected
  least squares, multinomial MLE, algorithm choices, scaling limits, and a
  source-linked validation plan.

AI disclosure: this organizational text was generated with AI assistance on
2026-08-17 and has not yet been independently verified. The linked theory
notes carry the same unverified-AI status until independent review.
