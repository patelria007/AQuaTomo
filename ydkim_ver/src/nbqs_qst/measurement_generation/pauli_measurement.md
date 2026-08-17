# `pauli_measurement.py` — Methods & Theoretical Background

> **AI disclosure:** This document was written and revised with AI assistance
> on 2026-08-17. It must not be marked verified until independently reviewed.

This document explains what each function in `pauli_measurement.py` does and
the theory it is built on. The shot-noise conventions follow the QST-literature
survey in [`theory_notes.md`](theory_notes.md) ("How QST Papers Inject Sampling Noise").

---

## Shared foundation: Pauli-basis tomography

The 4^n Pauli strings `P ∈ {I, X, Y, Z}^⊗n` form an **orthogonal operator basis**
for all 2^n × 2^n matrices under the Hilbert–Schmidt inner product
`Tr(A†B) = 2^n δ_AB`. Consequently, any density matrix is fixed by its Pauli
expectation values:

```
ρ = (1/2^n) · Σ_P  ⟨P⟩ P,      ⟨P⟩ = Tr(P ρ)
```

The module's whole purpose is to produce the numbers `⟨P⟩` — exactly (ideal),
or as finite-shot estimates (noisy) — since these are the direct input to
linear-inversion and MLE reconstruction.

---

## `pauli_strings(n)`

Enumerates all 4^n Pauli strings (`'II'`, `'IX'`, …, `'ZZ'`).
Theory: nothing deeper than the combinatorics of the Pauli basis above; the
ordering (`itertools.product`) is the canonical lexicographic one and is shared
by every other function, so results stay consistently keyed.

## `pauli_matrices(n, xp, device=None)`

Builds every Pauli operator as a `(4^n, 2^n, 2^n)` array.
When called from `pauli_expectations`, constants are created on `rho.device`;
standalone callers can provide the same device explicitly.
Theory: n-qubit Paulis are **tensor products** of one-qubit Paulis,
`P = P_1 ⊗ P_2 ⊗ … ⊗ P_n`. Implemented as n−1 **batched** broadcasting
outer products (extend the whole k-letter block at once with the four
one-qubit Paulis, then reshape): the naive per-string `xp.kron` chains
make 4^n·(n−1) tiny calls whose per-call dispatch overhead dominates the
runtime; the batched build is one order of magnitude faster and peaks at
~1× the final array size instead of the ~2× of a list + `stack` copy.
Tensor-product structure is what makes joint Pauli measurements local:
each qubit is measured individually in its own basis.

---

## `pauli_expectations(rho, shots=None)` — ideal path (`shots=None`)

Computes `⟨P⟩ = Tr(P ρ)` for all 4^n Paulis in one batched array operation:
`Tr(Pρ) = Σ_ij P_ij ρ_ji = Σ_ij P_ij (ρᵀ)_ij` — the elementwise product with
**ρ transposed**, summed.

Note on a tempting shortcut: summing the elementwise product with ρ itself
(`Σ_ij P_ij ρ_ij`) is valid only for *symmetric* Paulis (`Pᵀ = P`). While
I, X, Z are symmetric, **Y is skew-symmetric** (`Yᵀ = −Y`), so that shortcut
silently flips the sign of `⟨P⟩` for every Pauli containing an odd number of
Y's whenever ρ has complex entries (for real-valued ρ the error is invisible).
The transpose version above is correct for all Paulis and any ρ; a regression
test on the Y eigenstate `|+y⟩ = (|0⟩ + i|1⟩)/√2` (where `⟨Y⟩ = +1`) guards
against this.

Theory: the Born-rule expectation value of a Hermitian observable. No
probabilities are ever formed — this is the exact operator expectation, the
limit of infinitely many shots.

## `sample_outcomes(rho, basis, shots)` — the raw data generator

For one setting (a basis letter per qubit, e.g. `'XYZ'`):

1. Build the rotation `U = U_b1 ⊗ … ⊗ U_bn` (rows are the +1/−1
   **eigenbras** of each qubit's Pauli; X uses the Hadamard-like basis,
   Y uses (1, ∓i)/√2, Z is the identity/computational basis).
2. Compute the exact outcome distribution via the **Born rule** for projective
   measurement: `p(x) = ⟨x| U ρ U† |x⟩ = diag(U ρ U†)_x`, the diagonal of the
   state rotated into the measurement basis.
3. Draw `shots` **i.i.d. single-shot outcomes** from `p` (Style A, see survey):
   the snapshots are the experiment's raw data, and their finiteness *is* the
   shot noise — `Var(p̂_x) = p_x(1−p_x)/N`, binomial statistics exact at any N.

No Gaussian noise layer is added (it breaks at small N / rare outcomes — see
theory_notes.md survey).

## `generate_measurement_dataset(rho, shots, seed=...)`

Runs all `3^n` settings with one shared stdlib random stream and returns a
frozen `MeasurementDataset`. For each setting it retains both:

- the backend-native raw outcome vector of length `shots`, and
- a backend-native count vector of length `2^n`.

The aligned tuples `settings[i]`, `outcomes[i]`, and `counts[i]` describe one
experiment. This is the reconstruction-facing interface: linear inversion can
consume frequencies, while MLE can consume integer counts without resampling
or trying to reconstruct counts from rounded expectation values.

`expectations_from_dataset(dataset)` derives Pauli expectations from an
existing run, so analysis does not consume a second random stream.

## `pauli_expectations(rho, shots=N)` — noisy path

Measures each of the 3^n settings with N shots, then estimates every
**compatible** Pauli from those shots.

- *Compatibility:* Pauli `P` is estimated from setting `b` when each letter of
  `P` is `b`'s letter or `I` (e.g. `'XI'`, `'IY'`, `'XY'` from setting `'XY'`).
  Reason: the setting's projectors `U†|x⟩⟨x|U` are **joint eigenprojectors** of
  every compatible Pauli — one measurement setting diagonalizes them all.
- *Estimator:* on outcome bitstring `x`, the eigenvalue of `P` is the parity
  `(−1)^(# of 1-bits on P's support)`; the estimate is the sample mean of these
  ±1 parities:

  ```
  N_eff(P) = N · 3^(number of I letters in P)
  ⟨P⟩̂ = (1/N_eff) Σ_compatible settings Σ_shots
          (−1)^(popcount(x ∧ mask_P))                       (unbiased)
  Var(⟨P⟩̂) = (1 − ⟨P⟩²)/N_eff
  ```

  All compatible settings are pooled before division; later settings never
  overwrite earlier measurements. This is the textbook frequency estimator;
  `tests/test_measurement.py` exercises its `1/√N_eff` error law.
- *Shot economy:* 3^n settings cover all 4^n − 1 non-trivial Paulis because
  identity letters are read as **marginals** of the same shots. The
  all-identity string is the trivial observable, exactly 1.

---

## Backend compatibility (NumPy / CuPy / JAX / PyTorch)

The array namespace is taken from `rho` via `array_namespace(...)`, so the
same code runs on any backend whose array is passed in. Two rules keep
this true:

**Only operations present in every mainstream backend are used.** Almost
everything is array-API-standard (`asarray`, `arange`, `cumsum`, `sum`,
`where`, `zeros_like`, `full_like`, `real`, `matmul`, `matrix_transpose`,
`astype`, bitwise `>>`/`&`). Three helpers are not in the standard but exist in all four
backends and are relied on: `kron`, `diag`, `tolist`.

**Randomness: Python's stdlib, only.** The per-shot uniforms come
exclusively from `random.Random` — one code path, available everywhere,
seedable, and identical on every backend, so one `seed` reproduces the
same outcomes whether `rho` lives on NumPy, CuPy, JAX or PyTorch;
backend global RNGs are never touched. Only the CDF bucketing runs on
the array backend. The stdlib list is converted directly with `xp.asarray`;
there is no NumPy-specific bridge. Constants, uniforms, and count labels are
created on the input array's device rather than a backend's default device:

```python
est = pauli_expectations(rho, shots=1000)              # fresh stream (OS entropy)
est = pauli_expectations(rho, shots=1000, seed=0)      # seeded, reproducible everywhere
```

Caveats:

- **JAX defaults to 32-bit**, silently truncating the module's
  `complex128` requests. Enable `jax.config.update("jax_enable_x64", True)`
  before building `rho` for full precision (the backend test does this).
- **GPU sync points**: results are converted to Python floats/dicts on
  return, which forces a device→host sync. The exact path batches this
  into a single `tolist()`; the sampling path syncs once per estimated
  Pauli (a future vectorization could batch these too).

## Validation and numerical safeguards

Public functions reject non-square/non-power-of-two state shapes, non-finite
entries, non-Hermitian or non-positive-semidefinite matrices, a trace other
than one, invalid or mismatched basis strings, and non-positive/non-integer
shot counts with clear exceptions. Physicality is checked once per public
call, before any setting is sampled; this prevents probability clipping from
silently turning an invalid matrix into a different experiment. Before
sampling, tiny negative probability round-off is clipped and the distribution
is normalized. The inverse-CDF output is also bounded by `2^n - 1`, preventing
a last-bin overflow if the computed CDF ends infinitesimally below one.

The eigenspectrum validation costs `O(2^(3n))`, but a complete measurement run
already performs dense basis rotations for all `3^n` settings. The validation
cost is therefore accepted in exchange for a strict Born-rule input contract.

## Scaling limit

This is an exhaustive small-system QST simulator. The ideal path materializes
`4^n` dense `2^n × 2^n` Pauli matrices (`16^n` complex numbers), while the
sampling path visits all `3^n` settings and builds a dense basis rotation per
setting. It is therefore not a viable `n=20` full-tomography representation.
Larger systems require on-demand/tensor contractions or a reduced protocol
such as classical shadows; benchmark claims must state this limit explicitly.

---

## Internals

| Function | Theory / role |
|---|---|
| `_basis_rotation(basis, xp)` | Kronecker product of per-qubit eigenbra matrices — the basis-change unitary of the projective measurement. |
| `_compatible_expectation_sums(outcomes, basis, xp)` | Per-setting parity sums; the public dataset reducer pools these across every compatible setting. |
| `_outcome_counts(outcomes, num_outcomes, xp)` | Portable equality-and-sum count vector used instead of backend-specific `bincount`. |
| `_sample_categorical(probs, shots, xp, rng)` | Inverse-CDF sampling of a categorical distribution: one uniform per shot, bucketed through the cumulative distribution. Statistically equivalent to the Gumbel-max trick used in the QGOpt tutorial (see survey). |
| `_uniforms(shots, xp, rng)` | Per-shot uniforms from stdlib `random.Random`, converted directly with the detected array namespace. |
| `_setting_outcomes(rho, basis, shots, rng, xp)` | sample_outcomes with an explicit generator, so `pauli_expectations` can stream one rng across all its settings. |

---

## References

- Nielsen & Chuang, *Quantum Computation and Quantum Information* — Ch. 2
  (Pauli operators, projective measurement, Born rule), Ch. 8.4 (state tomography).
- James, Kwiat, Munro & White, "Measurement of qubits," Phys. Rev. A **64**, 052312 (2001) —
  estimating Pauli expectations from basis-measurement counts (linear inversion).
- Häffner, Roos & Blatt, "Quantum computing with trapped ions," Phys. Rep. **469**, 155 (2008) —
  circuit QST review; parity/frequency estimators from finite counts.
- Sampling-noise conventions (Style A): Torlai & Melko, arXiv:1703.05334;
  Schmale et al., npj QI **8**, 11 (2022);
  [QGOpt tomography tutorial](https://qgopt.readthedocs.io/en/latest/state_tomography.html) —
  detailed survey with links in [`theory_notes.md`](theory_notes.md).
