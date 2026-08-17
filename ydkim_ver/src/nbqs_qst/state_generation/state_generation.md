# `state_generation.py` — Methods, API, and Validation

> **AI disclosure:** This document and the associated module were generated
> with OpenAI Codex assistance on 2026-08-17. They remain **unverified** until
> independently reviewed, corrected where necessary, and validated.

This is the companion document for [`state_generation.py`](state_generation.py).
The longer literature review and derivations are in
[`theory_notes.md`](theory_notes.md).

## Scope

The module implements every state family required by the challenge in one core
Python file:

- independent local-Haar product pure states;
- global Haar-random pure states;
- induced Ginibre/Wishart random states with controllable rank and mean purity;
- depolarized random pure states with an exact target purity;
- deterministic GHZ and W reference states.

It also implements the challenge metrics purity, squared Uhlmann fidelity, and
pure-state overlap.

## Result object

Every generator returns a frozen `GeneratedState` record.

```python
from nbqs_qst.state_generation import random_haar_state

target = random_haar_state(like, 3, seed=17)
rho = target.rho       # pass directly to measurement_generation
ket = target.ket       # available for pure families
info = target.metadata()
```

`rho` is always present. `ket` is present only if the returned density matrix is
pure. The record also stores the family name, qubit count, seed, and relevant
parameters. This prevents a generated matrix from becoming separated from the
information needed to reproduce and interpret it.

## Backend selection and precision

Generation starts from a small prototype array named `like`:

```python
xp = array_namespace(like)
device = getattr(like, "device", None)
```

All returned arrays use this namespace and device. Core code does not import
NumPy. The same API therefore works with NumPy, CuPy, JAX, and PyTorch through
`array-api-compat`.

```python
import numpy as np
from nbqs_qst.state_generation import random_product_state

numpy_target = random_product_state(np.asarray(0.0), 4, seed=23)
```

For JAX, x64 must be enabled before creating `like`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

jax_target = random_product_state(jnp.asarray(0.0), 4, seed=23)
```

The output density matrices and kets use `complex128`. Tests compare NumPy and
JAX output for the same seed and also pass generated states through the Pauli
measurement simulator.

## Randomness contract

Every public random generator creates exactly one `random.Random(seed)` stream.
Real and imaginary Gaussian components are drawn from that stream in fixed
row-major order and only then converted to the requested backend. Backend RNGs
are never called.

A fixed seed therefore identifies the same mathematical sample on every
backend. Final floating-point bits may differ because reductions and matrix
multiplication can use different operation orders, so cross-backend dense
arrays are tested with strict `complex128` tolerances. The resulting fixed-seed
measurement outcomes are tested for exact equality.

## State families

### `random_product_state(like, n, seed=None)`

For each qubit, draw two independent complex Gaussian values and normalize:

$$
|\psi_j\rangle=\frac{(g_0,g_1)^T}{\sqrt{|g_0|^2+|g_1|^2}}.
$$

The global ket is

$$
|\Psi\rangle=\bigotimes_{j=0}^{n-1}|\psi_j\rangle.
$$

Each local state is uniform on the Bloch sphere, while the global state is
unentangled and has purity one. Tensor products proceed from qubit 0 to
qubit `n-1`, so qubit 0 is the most significant bit.

### `random_haar_state(like, n, seed=None)`

Draw a length-$d=2^n$ complex Gaussian vector and normalize it:

$$
|\psi\rangle=\frac{z}{\sqrt{z^\dagger z}}.
$$

The isotropic complex Gaussian direction has the unitary-invariant Haar
distribution. Constructing a full Haar unitary is unnecessary when only one
state vector is required.

### `random_mixed_state(like, n, k=None, seed=None)`

Draw a `d x k` complex Ginibre matrix and form

$$
\rho=\frac{GG^\dagger}{\operatorname{Tr}(GG^\dagger)}.
$$

The matrix is positive semidefinite and trace one by construction. It has rank
`min(d,k)` almost surely and ensemble-mean purity

$$
\mathbb E\operatorname{Tr}(\rho^2)=\frac{d+k}{dk+1}.
$$

The default `k=d` is the Hilbert--Schmidt ensemble. `k=1` is the Haar-pure
density-matrix ensemble; use `k>1` for a strictly mixed target. `k` controls a
distribution, not the exact purity of each sample.

### `random_state_with_purity(...)`

Given a Haar or product pure state `rho0`, construct

$$
\rho_\alpha=\alpha\rho_0+(1-\alpha)I/d,
\qquad
\alpha=\sqrt{\frac{\gamma-1/d}{1-1/d}}.
$$

Then `Tr(rho_alpha**2) = gamma` analytically for any
`gamma in [1/d, 1]`. This is a controlled benchmark family, not a uniform
fixed-purity ensemble and not an additional measurement-noise model.

### `ghz_state` and `w_state`

These deterministic states provide convention and structured-entanglement
checks:

$$
|\mathrm{GHZ}_n\rangle=(|0\ldots0\rangle+|1\ldots1\rangle)/\sqrt2,
$$

$$
|W_n\rangle=\frac{1}{\sqrt n}\sum_{j=0}^{n-1}|0\ldots010\ldots0\rangle.
$$

The W amplitudes occupy indices `2**(n-1-j)`, explicitly enforcing the
qubit-0-most-significant convention.

## Metrics

### `state_purity(rho)`

Computes

$$
\operatorname{Tr}(\rho^2)=\sum_{ij}\rho_{ij}\rho_{ji}
$$

with a transpose rather than a conjugating inner product.

### `pure_state_overlap(psi_true, psi_estimated)`

Computes the phase-invariant pure-state score

$$
|\langle\psi_{\rm true}|\psi_{\rm estimated}\rangle|^2.
$$

### `quantum_state_fidelity(rho, sigma)`

Implements the challenge's squared convention:

$$
F(\rho,\sigma)=
\left(\operatorname{Tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\right)^2.
$$

Hermitian eigendecompositions are used to construct the matrix square root.
Negative and numerically unresolved positive eigenvalues are clipped at a
dimension-scaled machine-precision threshold. This avoids a spurious
`sqrt(machine epsilon)` contribution for analytically rank-deficient states.

## Physical and scaling guarantees

The construction guarantees normalization and positivity analytically for
valid inputs. Tests additionally check Hermiticity, trace, numerical
eigenvalues, rank, purity, seed behavior, qubit ordering, and metric identities.

Creating a dense density matrix costs `O(4**n)` memory. The induced generator
also holds a `d x k` factor. These functions meet the exhaustive small-system
QST milestone; they do not make dense 20-qubit tomography practical. A future
large-system path should retain kets or low-rank factors rather than eagerly
forming `rho`.

## Test and figure commands

```bash
python -m pytest -q tests/test_state_generation.py
python -m pytest -q tests/test_measurement.py
python tests/analysis/generate_state_generation_figures.py
```

The plot script uses the public generators and stdlib-seeded samples. It does
not use a backend RNG. Output methodology is recorded in
[`docs/figures/state_generation/README.md`](../../../docs/figures/state_generation/README.md).

## References

- K. Życzkowski and H.-J. Sommers, "Induced measures in the space of mixed
  quantum states," *J. Phys. A* 34, 7111 (2001).
  [arXiv](https://arxiv.org/abs/quant-ph/0012101)
- F. Mezzadri, "How to generate random matrices from the classical compact
  groups," *Notices of the AMS* 54, 592 (2007).
  [arXiv](https://arxiv.org/abs/math-ph/0609050)
- D. N. Page, "Average Entropy of a Subsystem," *Phys. Rev. Lett.* 71,
  1291 (1993). [arXiv](https://arxiv.org/abs/gr-qc/9305007)
- W. Dür, G. Vidal, and J. I. Cirac, "Three qubits can be entangled in two
  inequivalent ways," *Phys. Rev. A* 62, 062314 (2000).
  [arXiv](https://arxiv.org/abs/quant-ph/0005115)

## Verification status

- AI-generated implementation and text: disclosed
- Automated tests: required before every delivery
- Independent human review: not yet recorded
- Current label: **UNVERIFIED — independent review required**
