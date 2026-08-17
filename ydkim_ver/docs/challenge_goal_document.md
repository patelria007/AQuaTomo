# Building a Hardware-Agnostic Quantum State Tomography Suite

### 2026 Niels Bohr Quantum Summer School
### SDU Odense, Denmark

---

## 1. Context and Motivation

A central challenge in quantum computing is verifying that a quantum system has actually prepared the state we intended to generate. Since we cannot directly observe a wavefunction, we must infer it using Quantum State Tomography (QST). By preparing the same state repeatedly, measuring it in different bases, and applying statistical reconstruction techniques, we estimate the underlying density matrix $\rho$.

In practice, quantum hardware is noisy and measurement samples are finite. Simple matrix inversion often produces non-physical results, such as matrices that are not positive semidefinite or have trace values inconsistent with a valid quantum state. This motivates the need to combine quantum theory, statistical inference, and numerical optimization.

As the number of qubits grows, the Hilbert space dimension increases exponentially, and the density matrix scales as $2^n \times 2^n$. Therefore, a practical tomography framework must not be tied to a single backend. It should run on CPUs, GPUs, and other accelerator hardware without hardcoded assumptions about the array library.

---

## 2. Problem Statement

The objective is to design and implement a modular Python package that evaluates how well different classes of randomly generated quantum states can be reconstructed under varying levels of measurement noise.

The key challenge is that the implementation must remain hardware-agnostic. The code must follow the Python Array API Standard (for example, via `array-api-compat`) and must not rely on NumPy-specific operations in core logic. Instead, functions should accept an array and detect its namespace (NumPy, CuPy, JAX, etc.) before performing calculations on the corresponding device.

---

## 3. Core Challenge

Your task is to build a complete software stack for QST that can:

1. generate representative quantum states,
2. simulate realistic measurement outcomes,
3. reconstruct the density matrix from noisy data,
4. compare reconstruction quality across algorithms and hardware backends.

---

## 4. Core Milestones

### 4.1 State Generation

Develop generators for distinct classes of $n$-qubit quantum states. At minimum, implement:

- Random Product States: states in which each qubit is independently randomized on the Bloch sphere and the overall state is unentangled.
- Haar-Random Pure States: states drawn uniformly from the Hilbert space, typically highly entangled.
- Random Mixed States: states with varying degrees of purity, where
  $$
  \mathrm{Tr}(\rho^2) < 1.
  $$
- Additional state families, if relevant to the research question.

Useful quantities for characterizing state quality include:

- purity:
  $$
  \gamma = \mathrm{Tr}(\rho^2),
  $$
- fidelity between two states $\rho$ and $\sigma$:
  $$
  F(\rho, \sigma) = \left(\mathrm{Tr}\sqrt{\sqrt{\rho}\,\sigma\,\sqrt{\rho}}\right)^2,
  $$
- and, for pure states, the overlap:
  $$
  |\langle \psi_{\text{true}} | \psi_{\text{est}} \rangle|^2.
  $$

---

### 4.2 Virtual Measurement Simulator

Construct a measurement simulator that emulates state preparation and readout in the standard Pauli bases $X$, $Y$, and $Z$.

For a given state $\rho$, the ideal measurement probabilities are

$$
 p_k = \mathrm{Tr}(E_k \rho),
$$

where $E_k$ denotes the measurement operator for outcome $k$.

If the experiment uses $N$ shots, then the observed counts follow a multinomial distribution:

$$
 (n_1, n_2, \ldots, n_m) \sim \mathrm{Multinomial}(N; p_1, p_2, \ldots, p_m).
$$

This produces noisy yet realistic finite-sample data, which is essential for evaluating tomography methods under realistic conditions.

---

### 4.3 Hardware-Agnostic Array API (The Compute Engine)

Instead of importing NumPy directly for all linear algebra, the core implementation should use the array namespace pattern:

```python
def calculate_fidelity(rho_true, rho_reconstructed):
    xp = array_namespace(rho_true, rho_reconstructed)
    # Use xp for all linear algebra and math
    return xp.trace(xp.linalg.matrix_power(...))
```

This ensures compatibility with NumPy, CuPy, JAX, and similar array libraries. The implementation must respect device-native execution and avoid hardcoded assumptions about the backend.

Important implementation note: functions should be written in a functionally pure style. In particular, avoid in-place updates such as `arr[0] = 1`, since JAX arrays are immutable by design.

---

### 4.4 State Reconstruction Engines

Implement at least two reconstruction strategies using the Array API framework.

#### 4.4.1 Linear Inversion

Use a pseudoinverse-based reconstruction scheme to map empirical measurement frequencies directly to an estimate of the density matrix. This is often the simplest approach, but it can produce unphysical states, including negative eigenvalues.

The linear inversion problem can be written schematically as

$$
 \hat{\rho} = A^{+} \mathbf{f},
$$

where:
- $\mathbf{f}$ is the vector of observed measurement frequencies,
- $A$ is the measurement matrix encoding the basis transformations,
- $A^{+}$ is the Moore–Penrose pseudoinverse.

This method is useful as a baseline, but it does not guarantee positivity or complete physical validity.

#### 4.4.2 Maximum Likelihood Estimation (MLE)

Implement an optimization routine to find the most likely physically valid state compatible with the measurement data. A common parameterization is via the Cholesky decomposition:

$$
 \rho = \frac{T^{\dagger} T}{\mathrm{Tr}(T^{\dagger} T)}.
$$

This guarantees that $\rho$ is positive semidefinite and normalized, provided $T$ is a valid matrix. The optimization seeks a density matrix $\rho$ that maximizes the likelihood of the observed measurement counts under the assumed measurement model.

---

## 5. Open-Ended Exploration

Once the core pipeline is functional, choose one or more research directions to investigate in depth.

### 5.1 Hardware Benchmarking

Scale the system size to $n = 3, 4, 5, \ldots, 20$ qubits and compare the runtime of the MLE reconstruction engine on CPU and GPU backends using the same codebase.

This can reveal how algorithmic scaling and backend choice affect performance in the high-dimensional regime.

### 5.2 Noise Resilience

Study how the required number of measurement shots scales as a function of target fidelity. For example, determine how many shots are needed to reach approximately $99\%$ fidelity for Haar-random states versus product states.

This question explores the balance between sample complexity and state complexity.

### 5.3 Advanced Protocols (Stretch Goal)

Implement a classical shadows protocol and compare its sample efficiency with that of standard QST for estimating specific observables.

This is an excellent extension for investigating how tomography can be made more efficient when only a limited set of observables matters.

---

## 6. Submission Requirements

Your final submission should be a self-contained Python software package featuring:

1. well-documented source code with modular, object-oriented design,
2. Array API compatibility across multiple backends,
3. a reproducible demonstration notebook or markdown documentation,
4. analysis of reconstruction quality and performance,
5. a summary of experimental findings and lessons learned.

---

## 7. Mandatory Disclosure of Generative AI Use

Participants must disclose any generative AI tools used in preparing the method, code, poster, or supplementary material. If AI tools were used, the contribution must be identified and the outputs must be independently checked, corrected, and validated.

This applies to:

- code generation,
- derivations or mathematical explanations,
- text generation,
- visualization assistance,
- documentation or presentation materials.

Undisclosed or unverified AI-generated claims may reduce the overall evaluation score.

---

## 8. Deliverables

Your final submission must include:

1. Well-documented Python source code using clean modular design and the Array API standard.
2. A markdown file or notebook demonstrating the complete pipeline on at least two backends (e.g., NumPy and CuPy or JAX).
3. A presentation of analytical findings, such as fidelity scaling plots and hardware benchmarking figures.
4. A reflection section describing failed attempts, unexpected observations, and what would be improved with more time.

---

## 9. Final Perspective

This challenge sits at the intersection of quantum information, numerical optimization, and high-performance computing. The goal is not only to reconstruct a state from measurement data, but to do so in a way that is physically meaningful, statistically sound, and computationally portable across hardware platforms.

A successful solution should therefore demonstrate both scientific rigor and engineering discipline: robust state modeling, realistic measurement simulation, accurate reconstruction, and scalable implementation across modern accelerator backends.
