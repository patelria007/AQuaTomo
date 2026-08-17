# `state_reconstruction.py` — methods and API

> **AI disclosure:** This implementation companion was generated with AI
> assistance on 2026-08-17. It remains unverified until a human independently
> checks the derivations, code, tests, and numerical results.

This document explains the single reconstruction implementation file
[`state_reconstruction.py`](state_reconstruction.py). Broader literature notes
and algorithm comparisons are in [`theory_notes.md`](theory_notes.md).

## Data contract

Every reconstruction consumes a
`measurement_generation.MeasurementDataset`. For `n` qubits it requires the
complete lexicographic setting tuple `{'X','Y','Z'}^n`, with one integer count
vector of length `d=2^n` per setting. Every count vector must be nonnegative
and sum to `shots_per_setting`.

Only `dataset.counts` are needed. Raw single-shot `dataset.outcomes` are not
re-read or resampled. Outcome index bits follow the measurement module:
qubit 0 is the most significant bit, `0` means Pauli eigenvalue `+1`, and `1`
means `-1`.

## Quick start

```python
from nbqs_qst.measurement_generation import generate_measurement_dataset
from nbqs_qst.state_reconstruction import reconstruct

dataset = generate_measurement_dataset(rho_true, shots=1000, seed=7)

linear = reconstruct(dataset, method="linear")
pls = reconstruct(dataset, method="pls")
mle = reconstruct(dataset, method="mle")

rho_est = mle.rho
print(mle.converged, mle.objective, mle.min_eigenvalue)
```

JAX must be configured before its arrays are created:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## `linear_inversion(dataset)`

The Pauli strings form an orthogonal operator basis:

$$
\rho=\frac{1}{2^n}\sum_{P\in\{I,X,Y,Z\}^{\otimes n}}
\langle P\rangle P.
$$

For every Pauli `P`, the implementation pools all settings compatible with
`P`. A count in outcome `x` contributes the parity

$$
\lambda_P(x)=\prod_{q:P_q\ne I}(-1)^{x_q}.
$$

The estimate is therefore

$$
\widehat\rho_{\rm LI}
=\frac{1}{2^n}\sum_P\widehat{\langle P\rangle}P.
$$

This estimate is Hermitian and trace one. It is intentionally **not** clipped
to be positive: finite-shot linear inversion can have negative eigenvalues,
and preserving this behavior is necessary for an honest baseline.

The Pauli matrices come from the measurement module, so reconstruction shares
its lexicographic Pauli ordering. Count parities are computed independently in
this module and are regression-tested against the measurement convention.

## `project_density_matrix(matrix)`

This function computes the Frobenius-nearest density matrix. It Hermitizes the
input, diagonalizes it, and projects its eigenvalue vector onto the probability
simplex:

$$
\mu_i=\max(\lambda_i-\tau,0),\qquad \sum_i\mu_i=1.
$$

It then reconstructs `V diag(mu) V†`. No in-place eigenvalue update is used,
so the operation works with immutable JAX arrays.

## `projected_least_squares(dataset)`

PLS is exactly

$$
\widehat\rho_{\rm PLS}=\mathcal P_{\mathcal D}
(\widehat\rho_{\rm LI}).
$$

It is a fast physical baseline and a useful MLE initializer. It is not called
MLE because the experiment uses multinomial shot noise rather than the special
additive-Gaussian model in which a single projection can equal an ML solution.

## `negative_log_likelihood(rho, dataset)`

For each setting `b` and outcome `x`, the implementation builds the tensor
product basis rotation `U_b` and evaluates

$$
p_{b,x}(\rho)=
\operatorname{diag}(U_b\rho U_b^\dagger)_x.
$$

The normalized objective is

$$
\mathcal C(\rho)=
-\frac{1}{N_{\rm total}}
\sum_{b,x:c_{b,x}>0}c_{b,x}\log p_{b,x}(\rho).
$$

Dividing by total shots changes gradient scale but not the optimum. Factorial
terms independent of `rho` are omitted. A zero-count term contributes zero.
If a positive-count outcome has probability at or below the documented
`probability_tolerance`, the public function returns infinity; it does not
silently add epsilon to the likelihood.

## `maximum_likelihood(dataset, ...)`

The MLE minimizes the exact multinomial NLL over

$$
\mathcal D=\{\rho:\rho=\rho^\dagger,\rho\succeq0,
\operatorname{Tr}\rho=1\}.
$$

The analytic matrix gradient is

$$
\nabla_\rho\mathcal C
=-\frac{1}{N_{\rm total}}
\sum_{b,x:c_{b,x}>0}\frac{c_{b,x}}{p_{b,x}}E_{b,x}.
$$

For one setting this is evaluated without materializing every projector:

$$
-U_b^\dagger\operatorname{diag}
\left(\frac{c_{b,x}}{N_{\rm total}p_{b,x}}\right)U_b.
$$

An iteration proposes

$$
\rho' = \mathcal P_{\mathcal D}
(\rho-\eta\nabla\mathcal C).
$$

Backtracking reduces `eta` until the proposal has finite probability for every
observed outcome and satisfies an Armijo objective-decrease condition. The
default initial state is `I/d`, which assigns strictly positive probability to
every Pauli outcome. `initial="pls"` and an explicit backend array are also
accepted. Convergence is declared when the state step is below tolerance or
when relative objective improvement remains below tolerance for five accepted
iterations; the latter handles flat likelihood directions without treating a
single small line-search step as convergence.

The algorithm returns a frozen `ReconstructionResult` containing:

- `rho`, `method`, `converged`, and accepted `iterations`;
- final normalized `objective` and monotone `objective_history`;
- `trace_error`, `hermiticity_error`, and `min_eigenvalue`.

`converged=False` must not be hidden by downstream analysis. Increasing the
iteration limit is not a substitute for checking the objective history and
physical diagnostics.

## `reconstruct(dataset, method=...)`

This convenience API returns the same `ReconstructionResult` shape for all
methods:

- `method="linear"`: closed-form LI, possibly nonphysical;
- `method="pls"`: projected LI;
- `method="mle"`: multinomial projected-gradient MLE.

## Quality metrics

`purity(rho)` implements the project definition

$$
\operatorname{Tr}(\rho^2).
$$

`state_fidelity(rho, sigma)` implements squared Uhlmann fidelity

$$
F(\rho,\sigma)=
\left(\operatorname{Tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\right)^2.
$$

`trace_distance(rho, sigma)` implements

$$
D(\rho,\sigma)=\frac12\|\rho-\sigma\|_1.
$$

The fidelity function is intended for physical density matrices. Applying it
to a nonphysical LI matrix and clipping the result would obscure LI's failure,
so poster analysis uses fidelity only for PLS/MLE and reports LI physicality
separately.

## Backend and numerical design

- Core logic imports no NumPy and obtains `xp` from backend-native counts.
- Constants are created with `complex128` on the count array's device.
- The same operations run on NumPy and JAX and are written for
  NumPy/CuPy/JAX/PyTorch Array API compatibility.
- There are no in-place updates and no backend RNG calls.
- Host conversions occur during input validation, simplex threshold selection,
  line-search decisions, and diagnostics. These are GPU synchronization points
  and must be included in performance interpretation.
- Measurement rotations are streamed setting by setting; all `6^n` projectors
  are never stored at once.

## Scaling limit

The implementation is exhaustive dense tomography. A density matrix contains
`4^n` complex numbers and one `complex128` matrix at `n=20` requires roughly
16 TiB. The present implementation therefore demonstrates correct portable
full QST only for small systems. A credible high-qubit extension needs a
low-rank/matrix-free/tensor representation or a reduced protocol such as
classical shadows; it cannot be obtained by merely moving this dense matrix to
a GPU.

## Validation

Run both reconstruction and upstream measurement tests:

```powershell
python -m pytest tests/test_reconstruction.py -q
python -m pytest tests/test_measurement.py -q
```

The reconstruction suite covers:

- exact `|+y>` recovery and the imaginary/Y sign;
- deliberate nonphysical LI and physical PLS;
- a one-qubit MLE with a known analytic symmetric solution;
- two-qubit Bell-state MLE fidelity;
- four-qubit GHZ reconstruction across all 81 Pauli settings;
- monotonic accepted NLL values and physical diagnostics;
- analytic-gradient versus central finite-difference agreement;
- reconstruction error proportional to `1/sqrt(shots)`;
- project metric definitions;
- fixed-count NumPy/JAX agreement;
- malformed dataset and option failures.

## Primary references

- Hradil, “Quantum-state estimation,” *Phys. Rev. A* 55, R1561 (1997),
  [DOI](https://doi.org/10.1103/PhysRevA.55.R1561).
- James et al., “On the Measurement of Qubits,” *Phys. Rev. A* 64, 052312
  (2001), [DOI](https://doi.org/10.1103/PhysRevA.64.052312).
- Shang, Zhang & Ng, “Superfast maximum-likelihood reconstruction for quantum
  tomography,” *Phys. Rev. A* 95, 062336 (2017),
  [arXiv](https://arxiv.org/abs/1609.07881).
- Bolduc et al., “Projected gradient descent algorithms for quantum state
  tomography,” *npj Quantum Information* 3, 44 (2017),
  [DOI](https://doi.org/10.1038/s41534-017-0043-1).
