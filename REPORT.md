# Building a Hardware-Agnostic Quantum State Tomography and Denoising Suite

## Technical report and implementation plan for the NBQSS 2026 challenge

Version 1.0 | 17 August 2026

## Abstract

This report presents an end-to-end solution strategy for the 2026 Niels Bohr Quantum Summer School challenge on hardware-agnostic quantum state tomography (QST). It analyzes the supplied attention-based denoising paper and state-generation notebook, identifies correctness and scaling issues, proposes a hierarchy of scalable alternatives to attention, and documents a working Python implementation. The implementation generates product, Haar-random pure, and random mixed states; applies physical noise channels; simulates finite-shot multinomial measurements across all local Pauli settings; reconstructs states by linear inversion and physical factorized maximum likelihood; applies training-free denoisers; computes fidelity and complementary diagnostics; persists data; and benchmarks the same kernels through the Python Array API pattern.

The main conclusion is that attention should not be the default denoiser. For dense few-qubit tomography, the strongest first line is a shot-noise-aware likelihood estimator combined with exact positive-semidefinite, trace-one constraints. In undersampled or nearly pure regimes, low-rank factorization or validated spectral shrinkage is more scalable and easier to audit than a general attention network. For genuinely larger systems, the representation must change: matrix-product states/operators and compressed sensing exploit structure, while classical shadows avoid reconstructing a full density matrix when only observables are needed. No algorithm can make generic full-state tomography polynomial in qubit count because the output itself has exponentially many degrees of freedom.

## 1. Executive recommendations

1. Use all 3^n tensor-product local Pauli settings for complete n-qubit Pauli tomography. The three global settings X^n, Y^n, and Z^n used in the supplied notebook are not informationally complete for n greater than one.
2. Replace ad hoc Gaussian perturbations of Pauli expectation values with multinomial sampling of physical outcome probabilities. Add device noise as a separate, explicit channel or measurement-confusion model.
3. Make exact density-matrix projection the mandatory physical baseline. Report raw linear inversion only to expose finite-shot pathologies.
4. Use factorized multinomial MLE, rho = T^dagger T / Tr(T^dagger T), as the main dense estimator. Cap the factor rank when a low-rank model is justified and validate that choice.
5. For pure or nearly pure states, compare low-rank spectral denoising and low-rank factorized likelihood before using a neural model. For unknown mixed states, use held-out likelihood to choose shrinkage strength rather than imposing rank one.
6. If the scientific question concerns a set of observables rather than the full state, switch to classical shadows. If the system is one-dimensional with limited entanglement, switch to a matrix-product representation.
7. Use attention only when there is repeatable, structured hardware noise that is not captured by a likelihood model, enough representative training data exist, and out-of-distribution and uncertainty tests are included.
8. Treat the Array API as a numerical-kernel contract, not as a claim that every backend has identical random-number, optimizer, or serialization APIs. Keep RNG and file I/O at explicit control-plane boundaries.

## 2. Challenge interpretation and requirement mapping

The challenge asks for a modular Python package that evaluates reconstruction quality for several random-state classes under varying measurement noise. Its defining constraint is backend neutrality: mathematical functions must discover an array namespace and execute on the array's native CPU, GPU, or accelerator backend. The required milestones are state generation, Pauli measurement simulation with finite shots, at least linear inversion and Cholesky-parameterized MLE, cross-backend demonstration, and analytical findings.

| Challenge requirement | Implemented component | Verification |
|---|---|---|
| Random product states | `random_product_state` | Hermiticity, unit trace, PSD, purity tests |
| Haar-random pure states | `haar_random_pure` | Unit trace and purity equal to one |
| Random mixed states | `random_mixed_state`, optional rank | PSD and purity below one |
| Pauli measurements | All settings in `{X,Y,Z}^n` | Exact one- and two-qubit inversion tests |
| Finite sampling | Seeded multinomial counts per setting | Counts sum exactly to shots per setting |
| Linear inversion | Pauli-expansion estimator | Exact recovery from exact probabilities |
| Physical MLE | Factorized multinomial MLE | PSD by construction; NLL monotonicity test |
| Hardware agnosticism | Namespace-discovered numerical kernels | NumPy and JAX smoke benchmark |
| Two-backend example | `examples/backend_smoke.py` | Matching two-qubit fidelity within 1.5e-8 |
| Analytical findings | CSV benchmark and this report | State-class, shot, physicality, and timing analysis |
| AI disclosure | README and Section 15 | Explicit contributions and validation steps |

The phrase “full stack” is used here in the scientific-computing sense: acquisition simulation, reconstruction, denoising, evaluation, persistence, command-line use, tests, and documentation. A browser application is not needed for the challenge and would distract from the numerical contract.

## 3. What the attention-based paper contributes

Palmieri et al. formulate QST post-processing as supervised matrix denoising. A standard estimator first maps finite-statistics data to a density matrix. The Cholesky factor of this noisy reconstruction is vectorized, passed through convolutional layers and a transformer layer, and mapped to the target Cholesky factor. Reconstructing the output as C C^dagger and normalizing the trace guarantees a physical state. The training objective combines a Cholesky-vector mean-square term with a trace-related regularizer. The paper compares neural post-processing of linear inversion and MLE and also tests out-of-distribution one-axis-twisting states with depolarizing and measurement/calibration noise.

The paper's important strengths are:

- It makes physicality part of the representation instead of hoping a neural output is PSD.
- It treats the initial QST estimator as a noisy observation and lets the network learn a conditional correction.
- It evaluates both in-distribution random states and physically motivated out-of-distribution states.
- It compares convolution-only and attention-based models at similar parameter counts.
- It explicitly targets few-degree-of-freedom experiments with appreciable unknown noise, rather than claiming generic many-body scalability.

Reported results include training sets of 2,000 pure states for MLE-NN and 5,000 for LI-NN in one benchmark. For d = 16 Haar-pure states with noiseless SIC-POVM probabilities followed by finite sampling, the paper reports transformer fidelities of 96.9%, 94.2%, and 81.1% at 10^5, 10^4, and 10^3 trials, compared with 94.2%, 87.0%, and 68.0% for the tested convolution-only alternatives. These results support attention when long-range correlations in a dense matrix representation matter.

The limitations are equally important:

- A supervised denoiser estimates a training-distribution conditional mean. Distribution shift can turn denoising into systematic bias.
- Cholesky coordinates are basis-dependent and unstable near rank-deficient states unless diagonal regularization is added.
- The architecture still consumes O(d^2) state parameters; reducing training examples does not remove exponential state dimension.
- Mean-square or Hilbert-Schmidt objectives need not align with the downstream observable, entanglement witness, coverage, or calibration target.
- Training cost, hyperparameter selection, and uncertainty are additional experimental resources.
- Neural improvement over MLE in a finite-sample regime does not imply asymptotic consistency or unbiasedness.

Therefore, the paper motivates a neural option but also suggests the correct baseline architecture: reconstruct physically, represent constraints explicitly, and validate on the target state family and noise process.

## 4. Audit of the supplied notebook

The notebook contains useful building blocks: Pauli matrices, tensor-product Pauli operators, three single-qubit eigenbases, a Pauli-expansion reconstruction, global depolarization, Cholesky vectorization, and generation of Haar-random target states. Several details must be corrected before the notebook can serve as challenge code.

### 4.1 Informational incompleteness

`all_pauli_bases(n)` constructs only the all-X, all-Y, and all-Z bases. For two qubits it misses XY, XZ, YX, YZ, ZX, and ZY; for n qubits it supplies 3 settings instead of 3^n. Those missing settings contain cross-axis correlations such as <X tensor Y>. A complete density matrix cannot be identified from the three global settings without a strong model assumption.

### 4.2 Noise is not finite-shot Pauli sampling

The notebook computes all 4^n Pauli expectation values and adds independent Gaussian noise divided by a fixed constant. This can produce values outside [-1,1], ignores covariance between outcomes in one setting, and does not make the noise scale correctly with each probability and shot count. A real finite-shot experiment samples a 2^n-outcome multinomial distribution for each chosen setting. Device noise should then be modeled independently, for example by a quantum channel before measurement or a calibrated confusion matrix after ideal probabilities.

### 4.3 The local depolarizing channel is not composed

Inside the loop over qubits, `new_density_matrix` is reset from the original state. The function therefore returns the effect of only the last loop iteration rather than a channel applied successively to every qubit. The corrected implementation updates the current state at each qubit and keeps the map trace-preserving and completely positive.

### 4.4 Eigenvalue “cleaning” is biased

`eigenvaluesCheck` uses a general eigensolver on a nominally Hermitian matrix, replaces negative eigenvalues with 0.0001, and renormalizes. This is not the nearest physical state and introduces an arbitrary full-rank floor. `PureEigenvaluesCheck` is more severe: values below approximately 0.99 are replaced by 0.0001, which effectively asserts near rank one even for mixed targets. The correct Frobenius projection hermitizes the input, diagonalizes with a Hermitian eigensolver, and projects the eigenvalue vector onto the probability simplex.

### 4.5 Rank-deficient Cholesky and dependency coupling

Pure states are rank deficient, so an ordinary Cholesky decomposition can fail unless a diagonal floor is introduced. The notebook converts NumPy arrays to Torch solely for Cholesky, while state generation and fidelity use Qiskit and QuTiP and noise uses SciPy. This makes backend movement implicit and blocks the Array API requirement. The new suite uses an eigen-factor initialization and keeps numerical kernels in the discovered namespace.

### 4.6 Reproducibility and software design

Paths are hard-coded to one user's machine; loops terminate through mutable counters; several imports and variables are unused; errors are silently skipped; and saved arrays mix input and target vectors without a schema. The replacement uses explicit seeds, exceptions with context, dataclasses for measurement data, a non-pickled NPZ manifest, a CLI, and unit tests.

## 5. Measurement and reconstruction model

For n qubits, d = 2^n. A local Pauli setting s = (s_1,...,s_n) belongs to {X,Y,Z}^n and has d bit-string outcomes b. Let U_s contain the tensor-product eigenvectors of the selected Pauli axes. The Born probability vector is

p_s = diag(U_s^dagger rho U_s).

For N shots per setting, the simulator draws

c_s ~ Multinomial(N, p_s).

The phrase “N shots” must always specify whether N is per setting or total. This implementation uses N per setting, so complete acquisition consumes N 3^n state copies.

### 5.1 Linear inversion

Every density matrix has the Pauli expansion

rho_LI = (1 / 2^n) sum over P in {I,X,Y,Z}^n of <P> P.

Each expectation is estimated from a compatible local setting. This estimator is unbiased before nonlinear physicality correction but can have negative eigenvalues at finite N. It is inexpensive, transparent, and an essential diagnostic.

### 5.2 Exact physical projection

Given a Hermitian estimate A = V diag(lambda) V^dagger, the nearest density matrix in Frobenius norm is obtained by projecting lambda onto the simplex {x_i >= 0, sum x_i = 1} and reconstructing with the same eigenvectors. Unlike elementwise clipping, the simplex threshold is chosen jointly so the result is the exact Euclidean projection.

### 5.3 Factorized multinomial MLE

The multinomial negative log likelihood is

L(rho) = - sum over settings s and outcomes b of c_(s,b) log p_(s,b)(rho).

The parameterization

rho(T) = T^dagger T / Tr(T^dagger T)

guarantees Hermiticity, PSD, and unit trace. T may be square or r by d. The latter is a Burer-Monteiro/rectangular Cholesky factor imposing rank at most r. The implementation computes the analytic likelihood gradient, uses backtracking, and accepts only non-increasing objective steps. This avoids a general optimizer dependency and works on Array API backends.

### 5.4 Metrics

No single metric is sufficient. The benchmark records squared Uhlmann fidelity, Hilbert-Schmidt distance, purity, minimum eigenvalue, and runtime. Fidelity is meaningful as a quantum-state similarity only when both inputs are physical. Raw linear inversion is therefore not ranked by fidelity when it has negative eigenvalues. For an experiment, add observable error, entanglement-witness error, calibration curves, bootstrap intervals, and coverage.

## 6. Denoising alternatives to attention

| Method | Training | Physical by construction | Dense cost | Best regime | Principal risk |
|---|---:|---:|---:|---|---|
| PSD trace-one projection | None | Yes | O(d^3) | Universal baseline | Bias at the PSD boundary |
| Factorized multinomial MLE | None | Yes | Iterative; dense O(d^3) scale | Trusted shot model, few qubits | Bias and local optimization at low samples |
| Low-rank spectral shrinkage | None | Yes | O(d^3), reducible with partial eigensolvers | Pure/nearly pure states | Severe underfitting of mixed states |
| Low-rank factored likelihood | None | Yes | Roughly O(r d^2) per dense step | Rank r much smaller than d | Rank misspecification |
| Held-out depolarizing shrinkage | None | Yes after projection | O(d^3) | Undersampled mixed states | Over-shrinkage if alpha is fixed |
| Compressed sensing | None | Yes in constrained variants | Structure-dependent | Approximately low-rank states, incomplete Pauli data | Optimization and measurement-design complexity |
| Matrix-product state/operator | Optional | Representation-dependent | Polynomial in n and bond dimension | 1D, local, limited entanglement | Fails for large bond dimension/long-range structure |
| Classical shadows | None | Not a full-state estimate | Polynomial for many observables | Observable prediction | Not a replacement for a requested full density matrix |
| CNN/residual denoiser | Supervised/self-supervised | Only with output factor | Lower parameter cost than attention | Local matrix/noise structure | Weak global correlation modeling |
| Score/diffusion prior | Substantial | With parameterization/projection | High inference cost | Rich multimodal prior | Training cost and hallucinated prior bias |

### 6.1 Best default: constrained statistical estimation

Shot noise is known, heteroscedastic, and coupled within a measurement setting. A likelihood method uses that information directly. It needs no training corpus, adapts to the actual shot count, and has interpretable failure modes. Add a calibrated readout channel or nuisance parameters when device noise is characterized. Use bootstrap resampling or likelihood-ratio regions for uncertainty.

### 6.2 Best pure-state scaling: low-rank factors

If the target is approximately rank r, store T as r by d rather than rho as d by d. This reduces parameters from O(d^2) to O(r d), and factored-gradient methods avoid repeated full PSD projection. Compressed-sensing theory shows that approximate low rank can reduce the number of Pauli measurements substantially. Rank must be selected by validation, information criteria, or a decreasing-spectrum diagnostic.

### 6.3 Best many-body scaling: tensor networks

For one-dimensional states generated by local dynamics and limited entanglement, a matrix-product state or operator represents the state with resources polynomial in n and bond dimension. Reconstruction can fit local marginals or measurement likelihoods without materializing a dense matrix. This is the most defensible route to “state tomography” beyond the dense few-qubit range, but it is a model of a structured state family, not generic tomography.

### 6.4 Best observable scaling: classical shadows

If the goal is energy, correlations, fidelity witnesses, or a collection of observables, full tomography is wasteful. Classical shadows use randomized measurements and can estimate many target properties with sample cost logarithmic in the number of requested properties, modulated by shadow norms. The output is an estimator for properties, not a physical dense rho.

### 6.5 When a neural denoiser is justified

A neural model becomes attractive when errors are structured, repeatable, difficult to parametrize, and represented in training data from the same device and calibration regime. Prefer the smallest architecture matching the structure: a residual MLP for a few coefficients, a CNN for local matrix patterns, a graph/tensor network for qubit connectivity, and attention only when long-range correlations measurably improve held-out performance. Use a Cholesky or low-rank output, train on multiple noise strengths, include physics and likelihood terms, and benchmark against validation-selected shrinkage and factorized MLE.

## 7. Hardware-agnostic architecture

The package separates five layers:

1. Domain layer: states, Pauli operators, channels, and physical constraints.
2. Acquisition layer: settings, Born probabilities, and multinomial counts.
3. Inference layer: inversion, projection, shrinkage, and factorized MLE.
4. Evaluation layer: metrics, experiment loops, summaries, and timing.
5. Control plane: random seeds, device transfer for unsupported multinomial RNG, NPZ/CSV I/O, and CLI parsing.

Numerical functions derive `xp` from input arrays. They avoid in-place updates, use functional accumulation, and create constants in the selected namespace. NumPy is mandatory only as the reference backend and control plane. CuPy and JAX are optional. The Python Array API Standard explicitly standardizes common array semantics and device support but does not promise identical backend random-number or runtime-switching systems; the explicit boundary is therefore part of portability, not an exception hidden from users.

The package layout is:

| Module | Responsibility |
|---|---|
| `backend.py` | Namespace discovery, device-aware conversion, adjoint, Kronecker product |
| `states.py` | Product, Haar, mixed, rank-controlled, and GHZ states |
| `operators.py` | Pauli strings and measurement unitaries |
| `noise.py` | Global and sequential local depolarization |
| `measurements.py` | Complete settings, Born probabilities, multinomial data |
| `reconstruction.py` | Pauli inversion and factorized MLE |
| `denoise.py` | Physical projection, low-rank and depolarizing shrinkage |
| `metrics.py` | Fidelity, distances, purity, physicality |
| `experiment.py` | Reproducible benchmark loops and summaries |
| `io.py`, `cli.py` | NPZ/CSV persistence and command-line workflows |

## 8. Experimental protocol

The included benchmark is a validation study, not a publication-scale claim. NumPy experiments use seed 7, n in {1,2}, 30 independent target states for each state class, and 100 or 500 shots per setting. Five estimators are applied to the identical measurement data. Factorized MLE uses at most 60 iterations. A smaller n = 3 scaling smoke test uses three states per condition and at most 30 MLE iterations. A second-backend JAX run uses five states per condition and 30 MLE iterations. Timing is eager wall-clock reconstruction time and is not a synchronized, warmed, JIT-compiled GPU benchmark.

State classes are product pure, Haar-random pure, and full-rank Ginibre/Hilbert-Schmidt mixed. Low-rank projection is intentionally fixed at rank one to expose both its benefit for pure states and its failure under rank mismatch. Depolarizing shrinkage uses alpha = 0.9; it is a baseline, not a claim that 0.9 is optimal. A production experiment should split shots and select alpha by held-out likelihood.

## 9. Results

### 9.1 Two-qubit reconstruction quality

The following values average 30 independent targets per cell. HS is Hilbert-Schmidt distance; lower is better.

| State | Shots/setting | Method | Mean fidelity | Mean HS |
|---|---:|---|---:|---:|
| Haar pure | 100 | Rank-1 projection | 0.9932 | 0.1092 |
| Haar pure | 100 | Factorized MLE | 0.9742 | 0.0925 |
| Haar pure | 500 | Rank-1 projection | 0.9985 | 0.0535 |
| Haar pure | 500 | Factorized MLE | 0.9891 | 0.0416 |
| Product pure | 100 | Rank-1 projection | 0.9932 | 0.1122 |
| Product pure | 100 | Factorized MLE | 0.9821 | 0.0870 |
| Product pure | 500 | Rank-1 projection | 0.9984 | 0.0548 |
| Product pure | 500 | Factorized MLE | 0.9890 | 0.0375 |
| Full-rank mixed | 100 | Rank-1 projection | 0.5731 | 0.5493 |
| Full-rank mixed | 100 | Factorized MLE | 0.9550 | 0.1466 |
| Full-rank mixed | 500 | Rank-1 projection | 0.5897 | 0.5191 |
| Full-rank mixed | 500 | Factorized MLE | 0.9874 | 0.0680 |

The metric ordering is informative. Rank-one projection gives the highest fidelity for pure targets, while MLE gives lower HS error. A pure reference makes fidelity especially sensitive to weight along the target vector and less sensitive to how residual weight is distributed, whereas HS penalizes the full matrix difference. For mixed targets, fixed rank one is decisively wrong. This is direct evidence that “best denoiser” is conditional on a scientifically defensible structural prior.

### 9.2 Physicality

For two-qubit Haar and product targets, linear inversion was nonphysical in 100% of the 30 replicates at both 100 and 500 shots per setting. For mixed states, it was nonphysical in 80% of replicates at 100 shots and 37% at 500 shots. All projection, shrinkage, and factorized-MLE outputs were physical to numerical tolerance. Raw linear-inversion fidelity was often close to or clipped at one despite negative eigenvalues, confirming that physicality must be checked before interpreting fidelity.

### 9.3 Runtime

Across all two-qubit NumPy conditions, mean reconstruction times were approximately 0.331 ms for linear inversion, 0.136 ms for exact physical projection, 0.121 ms for rank-one projection, 0.186 ms for shrinkage plus projection, and 62.0 ms for factorized MLE. These are implementation-level eager timings on one CPU, not universal performance constants.

The three-qubit smoke test shows the same qualitative behavior: linear inversion increases to about 1.4 ms and MLE to about 107 ms, while an eigendecomposition-based denoising step remains below 0.2 ms at d = 8. Only three replicates were used, so these values are directional.

### 9.4 Second-backend verification

The same two-qubit product-state smoke path produced fidelity 0.998599966 on NumPy and 0.998599981 on JAX, an absolute difference of about 1.5e-8. This validates namespace portability for the exercised path. Eager JAX was slower at these tiny dimensions because dispatch and compilation overhead dominate: mean two-qubit linear inversion was about 12.8 ms and MLE about 331 ms in the small run. This is not evidence against accelerators. A fair GPU/JAX study needs warm-up, synchronization, JIT boundaries, larger batches, the same iteration count, and device-resident data.

## 10. Interpretation: the recommended denoising stack

The practical stack should be staged rather than centered on one model.

1. Reconstruct raw linear inversion and record negative eigenvalues. This measures how far finite data push the estimate outside state space.
2. Project onto density-matrix space. This is the minimum credible physical baseline and is almost free at small d.
3. Fit full-rank factorized MLE with the correct likelihood. Compare held-out likelihood and target metrics.
4. If purity or a spectral gap is expected, fit several ranks and select by held-out likelihood or an information criterion. Do not infer “pure” only from a noisy rank-one-looking estimate.
5. Fit a shrinkage path rho_alpha = alpha rho_hat + (1-alpha) I/d and select alpha on held-out shots. This can beat unregularized MLE in undersampled HS risk, a phenomenon also analyzed in the supplied paper's Appendix I.
6. Add calibrated device noise to the forward model before learning an opaque correction.
7. Only then train a residual neural denoiser on measurement splits, multiple devices/noise levels, and out-of-distribution state families. Require it to improve held-out likelihood or downstream observable error, not only training-distribution fidelity.

## 11. Scaling roadmap

### Phase A: challenge-complete dense library

- Add readout-confusion matrices and general POVMs.
- Extend the implemented train/validation shot split to three-way and repeated cross-validation partitions.
- Add bootstrap confidence intervals and calibration tests.
- Add batched state/setting evaluation so accelerators receive sufficiently large kernels.
- Add synchronized, warmed NumPy/JAX/CuPy benchmarks and memory reporting.
- Add rank selection, early stopping, and optimizer diagnostics.

### Phase B: structured scalable estimators

- Implement rectangular rank-r factors throughout measurement prediction so rho is never materialized where possible.
- Add randomly subsampled Pauli settings and a compressed-sensing/factored-gradient reconstruction.
- Introduce matrix-product state/operator interfaces for one-dimensional systems.
- Add classical-shadow acquisition and observable estimators as a separate product surface.
- Generalize qubits to qudits and arbitrary local POVMs through operator-provider protocols.

### Phase C: hardware-agnostic library product

- Define backend capability checks and clear fallbacks for unsupported complex linear algebra.
- Add plugins/adapters for experiment records from Qiskit, Cirq, PennyLane, and vendor formats without making them core dependencies.
- Use immutable experiment manifests: state family, settings, shots, calibration version, backend, dtype, device, seed, and commit.
- Add continuous tests on NumPy CPU, JAX CPU/GPU, and CuPy GPU.
- Establish estimator interfaces with `fit`, `predict`, `diagnostics`, and uncertainty outputs.
- Version measurement schemas independently from numerical engines.

## 12. Failed attempts, surprises, and lessons

- The bundled environment initially lacked JAX, CuPy, and `array-api-compat`. A temporary JAX installation was used for validation rather than adding an accelerator as a mandatory package dependency.
- Small JAX jobs were slower than NumPy. This was expected after inspection: the benchmark is eager and dominated by dispatch overhead. Hardware agnosticism enables fair scaling tests; it does not guarantee speedup for tiny matrices.
- Fidelity alone made raw nonphysical linear inversion appear excellent. Minimum eigenvalue and HS error revealed the problem.
- Rank-one denoising was extremely effective for product/Haar pure states and extremely poor for full-rank mixed states. This sharp contrast is more useful than an average across all state classes.
- MLE often minimized HS error but did not maximize fidelity for pure targets under the chosen iteration budget. Metric choice and optimizer convergence both matter.
- The supplied notebook's Pauli-operator expansion was closer to a correct complete measurement design than its three-basis helper: it generated 4^n operator expectations, but the later Gaussian perturbation did not correspond to a realizable set of finite-shot measurement outcomes.

Given more time, the next priority would be automatic shrinkage/rank selection using the implemented held-out shot split, followed by batched accelerator kernels and uncertainty calibration. A neural comparison should come only after those baselines are established.

## 13. Validation and acceptance criteria

The implementation passes 12 unit tests covering state validity, measurement-setting completeness, multinomial count conservation, train/validation split conservation, exact inversion for one- and two-qubit states, rejection of incomplete data, physical projection, low-rank and shrinkage outputs, monotone MLE likelihood, noise-channel trace/PSD preservation, and NPZ round trips. The end-to-end example confirms negative eigenvalues in raw linear inversion, physical denoised outputs, high-fidelity recovery, and decreasing MLE objective.

Before using the suite for a scientific claim, require:

- At least two backends on the intended dtype/device, with numerical tolerance documented.
- Repeated seeds and confidence intervals, not only means.
- Separate state, shot-noise, channel-noise, and calibration-shift experiments.
- A predeclared primary metric and physicality checks.
- Out-of-distribution state families and noise strengths.
- Matched compute/measurement budgets across estimators.
- Bootstrap or likelihood-based uncertainty with coverage testing.
- Independent review of equations, code, and claims.

## 14. Reproducibility instructions

Install the package from `nbqst_suite` and run:

```text
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/end_to_end.py
nbqst benchmark --qubits 1 2 --shots 100 500 --states 30 --mle-iterations 60 --output results/final_benchmark.csv
```

For a second backend:

```text
JAX_ENABLE_X64=1 nbqst benchmark --backend jax --qubits 1 2 --shots 100 500 --states 5 --mle-iterations 30 --output results/jax_benchmark.csv
```

The included CSV files retain per-sample results and aggregated summaries. Shot counts are per setting. Seeds, state class, backend, physicality, fidelity, HS distance, purity, and timing are recorded.

## 15. Generative AI disclosure

OpenAI Codex was used to analyze the supplied challenge brief, paper, and notebook; propose the architecture and denoising comparison; derive and implement numerical routines; draft tests and documentation; run benchmarks; and draft this report. Principal AI-assisted code includes the Array API compatibility layer, complete local-Pauli simulator, physical projection, factorized MLE, metrics, CLI, tests, and report builder.

The contributions were independently checked within the project by exact analytical identities, automated tests, monotonic-likelihood checks, physicality diagnostics, two-backend numerical comparison, repeated seeded benchmarks, and visual inspection of the rendered report. These checks reduce but do not eliminate risk. The scientific interpretation, choice of experimental priors, and any publication claim require human domain review. The temporary JAX validation environment is not part of the delivered package.

## 16. References

1. Niels Bohr Quantum Summer School, “Building a Hardware-Agnostic Quantum State Tomography Suite,” challenge brief, 2026.
2. A. Macarone Palmieri et al., “Enhancing quantum state tomography via resource-efficient attention-based neural networks,” Physical Review Research 6, 033248 (2024), https://doi.org/10.1103/PhysRevResearch.6.033248.
3. Consortium for Python Data API Standards, “Python Array API Standard: Purpose and Scope,” https://data-apis.org/array-api/latest/purpose_and_scope.html.
4. D. Gross, Y.-K. Liu, S. T. Flammia, S. Becker, and J. Eisert, “Quantum state tomography via compressed sensing,” Physical Review Letters 105, 150401 (2010), https://arxiv.org/abs/0909.3304.
5. M. Cramer et al., “Efficient quantum state tomography,” Nature Communications 1, 149 (2010), https://arxiv.org/abs/1101.4366.
6. T. Baumgratz, D. Gross, M. Cramer, and M. B. Plenio, “Scalable reconstruction of density matrices,” New Journal of Physics 15, 125004 (2013), https://arxiv.org/abs/1207.0358.
7. H.-Y. Huang, R. Kueng, and J. Preskill, “Predicting many properties of a quantum system from very few measurements,” Nature Physics 16, 1050-1057 (2020), https://arxiv.org/abs/2002.08953.
8. C. Schwemmer et al., “Systematic errors in current quantum state tomography tools,” Physical Review Letters 114, 080403 (2015), https://doi.org/10.1103/PhysRevLett.114.080403.
9. M.-C. Hsu et al., “Quantum state tomography via nonconvex Riemannian gradient descent,” Physical Review Letters 132, 240804 (2024), https://doi.org/10.1103/PhysRevLett.132.240804.
10. K. Aditi and S. Becker, “Rigorous maximum-likelihood estimation for quantum states,” Physical Review A 112, 052436 (2025), https://doi.org/10.1103/j5gh-hmtw.
