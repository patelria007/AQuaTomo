# Theory Notes

> **AI disclosure:** This document was written and revised with AI assistance
> on 2026-08-17. It must not be marked verified until independently reviewed.

Collection of short summaries on theories/models used in this project, with sources.

**Index**
- [Shot Noise (Finite-Sampling Statistics)](#shot-noise-finite-sampling-statistics)
- [Readout Error (Measurement Assignment Error)](#readout-error-measurement-assignment-error)
- [How QST Papers Inject Sampling Noise (Survey)](#how-qst-papers-inject-sampling-noise-survey)

---

## Shot Noise (Finite-Sampling Statistics)

**What it is.** Statistical uncertainty in estimated probabilities that comes solely from using a finite number of measurement shots (N). It is not a physical noise channel acting on the state — it is sampling error on top of an (ideally noiseless) outcome distribution.

**Math.** For a single outcome with true probability `p`, the shot count `k` follows a binomial distribution:

```
k ~ Binomial(N, p)
p̂ = k / N            (MLE, unbiased)
Var(p̂) = p(1-p) / N
σ(p̂) = sqrt(p(1-p)/N)
```

- Poisson limit (`p << 1`): `σ(p̂) ≈ sqrt(p/N) = sqrt(k)/N` — the familiar "shot noise" `1/sqrt(N)` scaling.
- For a full outcome distribution `{p_x}`, the counts follow a multinomial, with covariance
  `Cov(p̂_x, p̂_y) = (δ_xy · p_x − p_x p_y) / N`.

**How to apply in simulation.** Given the ideal probability distribution `p` over outcomes:

1. **Sampling (standard):** draw `np.random.multinomial(N, p)` and use frequencies `n_x / N` as the estimated distribution.
2. **Analytic (Gaussian approximation, large N):** perturb `p` with Gaussian noise of covariance `(diag(p) − p pᵀ) / N`.

Shot noise combines multiplicatively with other errors: the noisy distribution is sampled `N` times, so it is applied *last*, after any physical noise channels.

**Sources**
- [Wikipedia — Shot noise](https://en.wikipedia.org/wiki/Shot_noise) (Poisson-noise view)
- [Photons, Shot Noise and Poisson Processes](https://www.strollswithmydog.com/photons-poisson-shot-noise/) (binomial ↔ Poisson connection)
- [arXiv:2501.03194 — Shots and variance on noisy quantum circuits](https://arxiv.org/html/2501.03194v1) (shot-number vs. variance in noisy circuits)
- [Clerk et al., Les Houches lecture notes — Quantum noise and quantum measurement](https://clerkgroup.uchicago.edu/PDFfiles/LesHouchesNotesAC.pdf) (broader measurement-noise background)

---

## Readout Error (Measurement Assignment Error)

**What it is.** A *classical* error in reporting the measurement result: the recorded bit string may differ from the true (ideal) outcome. It is characterized per-device by an **assignment probability matrix** (confusion matrix), calibrated by preparing known basis states and measuring.

**Math.** Define `A[m, n] = P(recorded n | true m)`. Rows sum to 1 (each row is a conditional distribution). Qiskit's convention is exactly this: `probabilities[m] = [P(0|m), ..., P(2^N − 1|m)]`.

The noisy outcome distribution `q` is the ideal distribution `p` pushed through the stochastic map:

```
q_n = Σ_m A[m, n] · p_m        (i.e. q = Aᵀ p)
```

**Special cases.**
- Single-qubit symmetric bit-flip with probability `p`:

  ```
  A = [[1-p,  p ],
       [ p , 1-p]]
  ```
- `N` independent qubits: `A = A_1 ⊗ A_2 ⊗ ... ⊗ A_N` (Kronecker product of per-qubit confusion matrices).

**How to apply in simulation.**
1. Compute the ideal outcome distribution `p` (e.g. from `Statevector.probabilities()`).
2. Map it through the confusion matrix: `q = Aᵀ p`.
3. Then draw shots from `q` (shot noise, see above) — i.e. **readout error first, shot noise second**.
4. (Mitigation, inverse problem: `p = (Aᵀ)⁻¹ q` when invertible, in practice via constrained least squares to keep probabilities valid.)

**Sources**
- [Qiskit Aer — ReadoutError (API docs)](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.ReadoutError.html) (matrix convention, normalization constraint, compose/tensor operations)
- [IBM Quantum — Build noise models](https://quantum.cloud.ibm.com/docs/guides/build-noise-models) (usage inside Aer noise models)
- [Qiskit (Medium) — Mitigate qubit measurement errors](https://medium.com/qiskit/mitigate-qubit-measurement-errors-in-qiskit-using-this-technique-1bb07ec319b5) (calibration + confusion-matrix mitigation)
- [PennyLane — Importing Qiskit noise models](https://www.pennylane.ai/demos/tutorial_how_to_import_qiskit_noise_models) (bit-flip readout error example)

---

## How QST Papers Inject Sampling Noise (Survey)

**Consensus.** In QST research, sampling noise is *not* added on top of data — it **is** the data generation process: draw i.i.d. single-shot outcomes from the exact Born-rule distribution of the target state. The dataset size (number of samples) then *plays the role* of the shot budget. Three implementation styles appear in the literature:

### Style A — i.i.d. single-shot snapshots (standard in ML-QST)
Sample raw outcome bit-strings from the exact distribution `p(x) = ⟨x|UρU†|x⟩` per measurement basis; train on the snapshots themselves (typically via negative log-likelihood).

- **Torlai & Melko (NN-QST, arXiv:1703.05334):** "exact sampling of the full wave-function |Ψ(σ)|²" (or QMC samples from it); e.g. 6400 samples per basis × 39 bases for N=20. No explicit noise model — reconstruction error is attributed entirely to "statistical uncertainty due to the finiteness of the training set". Emphasizes working on raw snapshots rather than averaged observables (no Gaussian stage at all).
- **Schmale et al. (CNN QST, npj QI 2022):** "compute a target density matrix exactly and compute its POVM distribution, from which we draw samples (1k–100k for 16-qubit systems)", using the Pauli-4 POVM (random x/y/z basis per qubit per shot). Physical noise (e.g. 3% dephasing) is baked into the *target state*, not the measurement; no Gaussian noise is added.
- **QGOpt tutorial:** 600,000 single-shot POVM outcomes drawn i.i.d. from `p_α = Tr(M_α ρ)` via the Gumbel-max trick (equivalent to categorical sampling).

### Style B — multinomial counts per measurement setting (linear inversion / MLE tomography)
Compute ideal probabilities for each setting, then draw one multinomial vector of counts per setting (`np.random.multinomial(N_shots, p)`); estimate frequencies. Used by framework-style tomography (Qiskit Experiments, QuTiP) and frequency-based estimators.

### Style C — Gaussian noise on estimated quantities (analytic treatments)
Replace sampled frequencies by `p̂ = p + N(0, Σ)` with multinomial covariance `Σ = (diag(p) − ppᵀ)/N`, or add `σ = sqrt((1 − ⟨A⟩²)/N)` noise directly to expectation values. Convenient for theory/scaling arguments, but the NN-QST community explicitly avoids it (raw snapshots carry strictly more information than the averaged estimates).

**Shot allocation conventions.**
- Fixed equal shots per basis (Torlai: `N_S` per basis over all 2N+1 or 3^N settings).
- Random basis per shot (Schmale: Pauli-4 POVM; same spirit as classical shadows).

**Takeaway for our simulation.** Inject sampling noise exactly as Style A: given the ideal outcome distribution per measurement setting, draw `N_shots` i.i.d. samples (equivalently one multinomial draw per setting), and never add a separate Gaussian noise layer on top.

**Status: implemented, awaiting independent verification.**
`pauli_measurement.py` follows Style A: `sample_outcomes(rho, basis, N)` draws
N i.i.d. single-shot outcomes from the exact Born distribution per setting
(no multinomial shortcut, no Gaussian layer). `generate_measurement_dataset`
retains raw outcomes and count vectors for every setting, and
`pauli_expectations(rho, shots=N)` pools the ±1 parity observations from every
compatible setting. Per-function theory: see `pauli_measurement.md`.

**Sources**
- [Torlai & Melko — Many-body quantum state tomography with neural networks (arXiv:1703.05334)](https://arxiv.org/abs/1703.05334) (also [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1703.05334))
- [Torlai et al. — Neural-network quantum state tomography, Nat. Phys. 14, 447 (2018)](https://www.nature.com/articles/s41567-018-0048-5) + [reference code (nnqsr)](https://github.com/GTorlai/nnqsr)
- [Schmale et al. — Efficient quantum state tomography with CNNs, npj QI 8, 11 (2022)](https://www.nature.com/articles/s41534-022-00621-4)
- [QGOpt docs — Quantum state tomography tutorial](https://qgopt.readthedocs.io/en/latest/state_tomography.html)
- [QuTiP / numerical QST simulation discussion](https://quantumcomputing.stackexchange.com/questions/5025/numerical-quantum-state-tomography-simulator)
