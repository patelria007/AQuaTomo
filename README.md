# NBQST hardware-agnostic tomography suite

This repository is a dependency-light reference implementation for the 2026
Niels Bohr Quantum Summer School challenge, “Building a Hardware-Agnostic
Quantum State Tomography Suite.” It covers the complete path from state
generation to finite-shot local-Pauli data, physical reconstruction, denoising,
metrics, persistence, and reproducible benchmarks.

## What is implemented

- State ensembles: random product, Haar-random pure, Hilbert-Schmidt/Ginibre
  mixed, rank-controlled mixed, and GHZ states.
- Noise: global and sequential local depolarizing channels.
- Measurements: all `3**n` tensor-product Pauli settings with `2**n` outcomes
  per setting, true multinomial shot counts, and optional independent readout
  errors.
- Reconstruction: Pauli linear inversion and multinomial maximum likelihood
  with the physical factorization `rho = T^dagger T / trace(T^dagger T)`.
- Training-free denoisers: exact Frobenius projection to the density-matrix
  simplex, low-rank spectral thresholding, and shrinkage toward `I/d`.
- Metrics: squared Uhlmann fidelity, Hilbert-Schmidt distance, trace distance,
  purity, and minimum eigenvalue.
- Full workflow: NPZ data bundles, CSV benchmarks, command-line tools, examples,
  and unit tests.
- Validation tools: hypergeometric train/validation shot splitting and
  held-out-likelihood selection of depolarizing shrinkage.

The notebook supplied with the challenge measures only `XXX...`, `YYY...`, and
`ZZZ...`. Those three global settings are not informationally complete once
`n > 1`; the package therefore uses all local Pauli strings over `X/Y/Z`.

## Installation

From this directory:

```bash
python -m pip install -e .
```

NumPy is the only mandatory dependency. For stricter cross-backend namespace
normalization, install `array-api-compat`; CuPy and JAX remain optional:

```bash
python -m pip install -e '.[array-api,test]'
```

## Quick start

```bash
python examples/end_to_end.py

nbqst generate --qubits 2 --shots 1000 --samples 20 \
  --state-type haar --output data/haar2.npz

nbqst reconstruct data/haar2.npz --rank 1 \
  --output results/haar2_reconstruction.csv

nbqst benchmark --qubits 1 2 3 --shots 100 500 2000 \
  --states 10 --output results/benchmark.csv
```

To generate data with asymmetric per-qubit readout fidelity:

```bash
nbqst generate --qubits 2 --shots 1000 --samples 20 \
  --readout-fidelity-0 0.98 0.97 --readout-fidelity-1 0.96 0.95 \
  --output data/haar2_readout.npz
```

Here `readout-fidelity-0` is `P(measured 0 | true 0)` and
`readout-fidelity-1` is `P(measured 1 | true 1)`. A single value is broadcast
to every qubit. The confusion model is applied to Born probabilities before
multinomial sampling; reconstruction remains intentionally uncorrected.

To exercise an available accelerator without changing numerical code:

```bash
nbqst benchmark --backend cupy --qubits 1 2 --shots 100 1000
nbqst benchmark --backend jax  --qubits 1 2 --shots 100 1000
python examples/backend_smoke.py
```

## Array API design

Every linear-algebra kernel discovers `xp` from input arrays. The code avoids
in-place mutation, so JAX immutability is respected. Device arrays remain on
their native device for Born probabilities, reconstruction, eigendecomposition,
matrix products, likelihood gradients, and metrics.

Multinomial RNG, CSV/NPZ files, and convergence logging are explicit
control-plane operations. Multinomial sampling is not part of the Python Array
API Standard; probabilities are copied to a seeded NumPy generator and sampled
counts are immediately copied back to the original backend/device. This
boundary is deliberate and documented rather than hidden in numerical kernels.

## Denoising guidance

Use methods in this order:

1. Always compare against raw linear inversion and exact physical projection.
2. Prefer factorized multinomial MLE when the full density matrix is still
   tractable and the measurement likelihood is trusted.
3. For nearly pure states, validate a low-rank factor or spectral truncation.
4. In the undersampled regime, select depolarizing shrinkage on held-out shots.
5. For larger systems, change the representation: low-rank factors,
   matrix-product states/operators, or classical shadows for selected
   observables. No dense denoiser can remove the exponential output size of a
   generic `2**n x 2**n` density matrix.

Attention networks may still be useful for unknown, repeatable device noise,
but they should be compared to these physics-constrained baselines, evaluated
out of distribution, and accompanied by uncertainty and calibration checks.

## Testing

```bash
python -m unittest discover -s tests -v
```

The tests cover state physicality, measurement completeness, exact one- and
two-qubit inversion, negative-eigenvalue removal, MLE monotonicity, noise
channels, serialization, and a command-line smoke path.

## Scaling

Dense full-state tomography is exponential by definition: the density matrix
has `4**n` real degrees of freedom. This reference package deliberately makes
that cost visible. Complete local Pauli acquisition has `3**n` settings, dense
eigendecomposition costs `O(8**n)` time and `O(4**n)` memory, and MLE repeats
Born calculations over all settings. Use the dense engines for validation and
small systems (typically up to roughly 5-8 qubits depending on hardware), then
switch to structural models or observable-only protocols.

## Reproducibility and AI disclosure

All examples use explicit seeds. Benchmark CSV files include backend, state
class, shot count, estimator, fidelity, physicality, and wall time. This initial
implementation and accompanying report were drafted with OpenAI Codex. The
principal AI-assisted contributions were software architecture, mathematical
derivations, implementation, tests, and report drafting. They were checked by
unit tests, exact-state identities, negative-log-likelihood monotonicity,
cross-method comparisons, and visual document review. Independent scientific
review and larger statistical replication remain required before publication.
