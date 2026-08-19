# NBQST hardware-agnostic tomography suite

This repository is a dependency-light reference implementation for the 2026
Niels Bohr Quantum Summer School challenge, “Building a Hardware-Agnostic
Quantum State Tomography Suite.” It covers the complete path from state
generation to finite-shot local-Pauli data, physical reconstruction, denoising,
metrics, persistence, and reproducible benchmarks.

## What is implemented

- State ensembles: random product, Haar-random pure, Hilbert-Schmidt/Ginibre
  mixed, rank-controlled mixed, and GHZ states.
- Noise: global/local depolarizing, amplitude damping, phase damping,
  asymmetric Pauli, coherent rotation, and asymmetric readout confusion.
- Measurements: all `3**n` tensor-product Pauli settings with `2**n` outcomes
  per setting and true multinomial shot counts.
- Reconstruction: Pauli linear inversion and multinomial maximum likelihood
  with the physical factorization `rho = T^dagger T / trace(T^dagger T)`.
- Neural reconstruction: a separately trained, feed-forward ReLU/tanh network
  that maps Pauli frequencies to a Cholesky factor and therefore always returns
  a positive semidefinite, trace-one density matrix. Training and inference use
  only the mandatory NumPy dependency; inference follows the input array backend.
- Training-free denoisers: exact Frobenius projection to the density-matrix
  simplex, low-rank spectral thresholding, and shrinkage toward `I/d`.
- Metrics: squared Uhlmann fidelity, Hilbert-Schmidt distance, trace distance,
  purity, and minimum eigenvalue.
- Full workflow: NPZ data bundles, CSV benchmarks, command-line tools, examples,
  and unit tests.
- Object-oriented facade: `TomographyPipeline` provides a compact application
  API while retaining the separately testable functional numerical kernels.
- Classical shadows: randomized local-Pauli acquisition, reusable observable
  queries, median-of-means aggregation, empirical uncertainty, and a direct
  targeted-Pauli reference estimator.
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
python examples/neural_comparison.py

nbqst generate --qubits 2 --shots 1000 --samples 20 \
  --state-type haar --output data/haar2.npz

nbqst reconstruct data/haar2.npz --rank 1 \
  --output results/haar2_reconstruction.csv

nbqst benchmark --qubits 1 2 3 --shots 100 500 2000 --states 10
```

For the smallest annotated end-to-end recipe, open and run
`notebooks/NBQSS_Backend_Quickstart.ipynb`. It defaults to NumPy and contains
the exact one-line backend switch for CuPy or JAX.

## Classical shadows for observables

One local-Pauli shadow data set can be queried for many Pauli observables after
measurement:

```python
from nbqst.shadows import ClassicalShadowProtocol, observable_expectation

protocol = ClassicalShadowProtocol(median_of_means_groups=5)
shadow = protocol.acquire(rho, 5000, rng=7)
for estimate in protocol.estimate_many(shadow, ("ZII", "IZZ", "XYZ")):
    exact = observable_expectation(rho, estimate.observable)
    print(estimate.observable.label, exact, estimate.value, estimate.standard_error)
```

For a weight-`k` Pauli string, a random local basis matches its support with
probability `3**(-k)`. This makes the protocol most useful for collections of
low-weight observables. If one observable is fixed before acquisition, measure
that Pauli setting directly instead; the extended study includes both the
single-target best case and a fair split-budget multi-observable baseline.

The implementation follows Huang, Kueng, and Preskill, “Predicting many
properties of a quantum system from very few measurements,” Nature Physics 16,
1050–1057 (2020), arXiv:2002.08953.

To exercise an available accelerator without changing numerical code:

```bash
nbqst benchmark --backend cupy --qubits 1 2 --shots 100 1000
nbqst benchmark --backend jax  --qubits 1 2 --shots 100 1000
python examples/backend_smoke.py
```

## Synchronized LI/MLE/NN timing

Train or supply one dimension-specific neural model for every requested qubit
count. Training is outside the inference timer:

```bash
python examples/neural_comparison.py --qubits 1 --shots 500 \
  --output results/neural_1q.csv
python examples/neural_comparison.py --qubits 2 --shots 500 \
  --output results/neural_2q.csv
```

Then run the same grid on each backend. The default output name includes the
backend, so successive NumPy, CuPy, and JAX runs do not overwrite one another:

```bash
nbqst benchmark --backend numpy --qubits 1 2 --shots 100 1000 \
  --state-types product haar mixed --states 5 \
  --methods li mle nn \
  --neural-model 1=results/neural_1q_model.npz \
  --neural-model 2=results/neural_2q_model.npz \
  --mle-iterations 60 --warmup-rounds 2 --timing-repeats 5

nbqst benchmark --backend cupy --qubits 1 2 --shots 100 1000 \
  --state-types product haar mixed --states 5 \
  --methods li mle nn \
  --neural-model 1=results/neural_1q_model.npz \
  --neural-model 2=results/neural_2q_model.npz \
  --mle-iterations 60 --warmup-rounds 2 --timing-repeats 5

JAX_ENABLE_X64=1 nbqst benchmark --backend jax \
  --qubits 1 2 --shots 100 1000 \
  --state-types product haar mixed --states 5 \
  --methods li mle nn \
  --neural-model 1=results/neural_1q_model.npz \
  --neural-model 2=results/neural_2q_model.npz \
  --mle-iterations 60 --warmup-rounds 2 --timing-repeats 5
```

Each run writes a detailed CSV, a grouped summary CSV, and a JSON manifest.
Recorded intervals use `perf_counter_ns` with a host barrier, CuPy current-stream
synchronization, or JAX `block_until_ready`, as appropriate. Reconstruction and
fidelity are timed separately. MLE includes its LI/physical initialization;
neural weights are copied to the selected backend before warm-up. Set
`--warmup-rounds 0` only when deliberately measuring cold-start/compilation.

Raw LI may be nonphysical. Its row therefore includes `is_physical` and
`fidelity_interpretable`; do not rank a nonphysical LI estimate by fidelity.

Merge the three backends, calculate median NumPy-relative speedups, and create
state-family scaling plots with:

```bash
python tools/compare_backend_timings.py \
  --inputs results/benchmark_numpy.csv results/benchmark_cupy.csv results/benchmark_jax.csv \
  --output results/backend_timing_summary.csv \
  --plot-dir results/backend_timing_plots
```

Install `.[plot]` if plots are requested. Keep qubits, shots, state seeds,
models, MLE iterations, warm-ups, timing repeats, precision, and scheduler
allocation identical across backends.

## Reproduce the analytical extension

The following creates synchronized hardware timings, a discrete shot-grid
search for 99% fidelity, a matched-budget observable study, plots, raw CSVs,
and an execution manifest:

```bash
python tools/run_extended_study.py --hardware-qubits 1 2 3 4 5
```

The dense resource table always extends through 20 qubits. Empirical timing is
written only for backends that are actually installed and executable; missing
CuPy/JAX packages are recorded as unavailable rather than replaced by inferred
GPU data. See `results/extended_study/` and `ANALYTICAL_FINDINGS.md`.

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

The included neural estimator is not a challenge requirement. It is an
independent implementation motivated by D. Koutny et al., "Neural-network
quantum state tomography" (arXiv:2206.06736), and is provided so linear
inversion, maximum likelihood, and learned reconstruction can be evaluated on
identical simulated measurements. The example intentionally reports neural
training cost separately from per-state inference cost.

## Testing

```bash
python -m pytest -q
python tools/run_verification_study.py --max-scaling-qubits 7
```

The 23 tests cover state physicality, measurement completeness, exact one- and
two-qubit inversion, negative-eigenvalue removal, MLE monotonicity, noise
channels, serialization, synchronized three-method timing records, manifests,
the object-oriented facade, classical-shadow acquisition and estimators, and a
command-line smoke path.

The staged verification study additionally checks the Haar overlap law,
analytic channel action, exact LI through five qubits, LI/MLE/NN behavior under
seven noise cases, physicality, and dense scaling.  It writes all figures,
tables, pass/fail gates, and the execution manifest to
`results/verification_study/`.  Use `CLUSTER_VERIFICATION.md` for the CPU and
GPU-cluster rerun procedure and the required hardware/job metadata.

## Scaling

Dense full-state tomography is exponential by definition: the density matrix
has `4**n` real degrees of freedom. This reference package deliberately makes
that cost visible. Complete local Pauli acquisition has `3**n` settings, dense
eigendecomposition costs `O(8**n)` time and `O(4**n)` memory, and MLE repeats
Born calculations over all settings. Use the dense engines for validation and
small systems (typically up to roughly 5-8 qubits depending on hardware), then
switch to structural models or observable-only protocols.

On the documented Apple M1 CPU smoke run, the largest completed dense case was
seven qubits (2,187 settings); the capped eight-qubit acquisition did not finish
within 45 seconds.  These are local reproducibility observations, not universal
limits or accelerator claims.

The experiment owner subsequently confirmed that both commands below completed
on one cluster node with one NVIDIA RTX 4060 Ti, PCIe 4, 8 GB, and `sm 8.7`
(recorded exactly as supplied):

```bash
nbqst benchmark --backend cupy --qubits 1 2 --shots 100 1000
nbqst benchmark --backend jax  --qubits 1 2 --shots 100 1000
```

This confirms CuPy/CUDA and JAX backend execution, not an accelerator speedup.
No benchmark output or timing table was supplied.  See
`results/verification_study/accelerator_execution_confirmation.json` for the
confirmed and still-missing provenance fields.

## Reproducibility and AI disclosure

All examples use explicit seeds. Benchmark CSV files include backend, state
class, shot count, estimator, fidelity, physicality, and wall time. This initial
implementation and accompanying report were drafted with OpenAI Codex. The
principal AI-assisted contributions were software architecture, mathematical
derivations, implementation, tests, and report drafting. They were checked by
unit tests, exact-state identities, negative-log-likelihood monotonicity,
cross-method comparisons, and visual document review. Independent scientific
review and larger statistical replication remain required before publication.
