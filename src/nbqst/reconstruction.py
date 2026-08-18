"""Linear inversion and physical factorized maximum-likelihood tomography."""

from __future__ import annotations

from .backend import adjoint, array_namespace, asarray, complex_dtype, device_of, eye, scalar, zeros
from .denoise import hermitize, project_density_matrix
from .measurements import MeasurementData, complete_pauli_settings, pauli_probabilities
from .operators import measurement_unitary, pauli_labels, pauli_string


def _canonical_setting(pauli_label: str) -> str:
    return "".join("Z" if ch == "I" else ch for ch in pauli_label)


def _parity_signs(pauli_label: str, xp, *, device=None):
    n_qubits = len(pauli_label)
    dim = 2**n_qubits
    active = tuple(i for i, ch in enumerate(pauli_label) if ch != "I")
    signs = []
    for outcome in range(dim):
        parity = sum((outcome >> (n_qubits - 1 - i)) & 1 for i in active) % 2
        signs.append(1.0 if parity == 0 else -1.0)
    return asarray(signs, xp, dtype=getattr(xp, "float64", None), device=device)


def linear_inversion_pauli(data: MeasurementData):
    """Reconstruct by the n-qubit Pauli expansion.

    Complete data require all 3^n local Pauli settings.  Identity positions use
    a deterministic Z setting; this avoids redundant averaging and makes the
    estimator easy to audit.
    """

    required = set(complete_pauli_settings(data.n_qubits))
    missing = required - set(data.settings)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Linear inversion requires all 3^n local settings; missing {len(missing)} ({preview})")
    first = next(iter(data.counts.values()))
    xp = array_namespace(first)
    device = device_of(first)
    dim = 2**data.n_qubits
    rho = zeros((dim, dim), xp, dtype=complex_dtype(xp), device=device)
    for label in pauli_labels(data.n_qubits):
        if set(label) == {"I"}:
            expectation = asarray(1.0, xp, dtype=getattr(xp, "float64", None), device=device)
        else:
            setting = _canonical_setting(label)
            frequencies = data.counts[setting] / xp.sum(data.counts[setting])
            expectation = xp.sum(frequencies * _parity_signs(label, xp, device=device))
        rho = rho + expectation * pauli_string(label, xp)
    return hermitize(rho / dim)


def negative_log_likelihood(rho, data: MeasurementData, *, epsilon=1e-12) -> float:
    total = 0.0
    count_sum = 0.0
    xp = array_namespace(rho)
    for setting, counts in data.counts.items():
        probabilities = pauli_probabilities(rho, setting)
        probabilities = xp.maximum(probabilities, xp.asarray(epsilon, dtype=probabilities.dtype))
        total = total - xp.sum(counts * xp.log(probabilities))
        count_sum = count_sum + xp.sum(counts)
    return scalar(total / count_sum)


def _factor_to_density(factor, xp):
    gram = adjoint(factor, xp) @ factor
    return hermitize(gram / xp.real(xp.trace(gram)))


def _initial_factor(rho, xp, rank=None):
    values, vectors = xp.linalg.eigh(project_density_matrix(rho))
    dim = rho.shape[-1]
    rank = dim if rank is None else int(rank)
    if not 1 <= rank <= dim:
        raise ValueError(f"rank must be between 1 and {dim}")
    values = xp.maximum(xp.real(values[-rank:]), xp.asarray(1e-12, dtype=values.dtype))
    vectors = vectors[:, -rank:]
    # T has shape (rank, dim) and rho = T^dagger T / Tr(T^dagger T).
    return xp.sqrt(values)[:, None] * adjoint(vectors, xp)


def _likelihood_gradient(rho, data, xp, epsilon):
    dim = rho.shape[-1]
    gradient = zeros((dim, dim), xp, dtype=rho.dtype, device=device_of(rho))
    total_counts = asarray(0.0, xp, dtype=getattr(xp, "float64", None), device=device_of(rho))
    for setting, counts in data.counts.items():
        unitary = measurement_unitary(setting, xp)
        probabilities = pauli_probabilities(rho, setting)
        probabilities = xp.maximum(probabilities, xp.asarray(epsilon, dtype=probabilities.dtype))
        weights = counts / probabilities
        gradient = gradient - (unitary * weights[None, :]) @ adjoint(unitary, xp)
        total_counts = total_counts + xp.sum(counts)
    return hermitize(gradient / total_counts)


def factorized_mle(
    data: MeasurementData,
    *,
    initial=None,
    rank=None,
    max_iter: int = 250,
    learning_rate: float = 0.25,
    tolerance: float = 1e-9,
    epsilon: float = 1e-12,
    return_history: bool = False,
):
    """Multinomial MLE with ``rho = T^dagger T / Tr(T^dagger T)``.

    The factor is a rectangular Cholesky/Burer-Monteiro factor.  Rank can be
    capped for scalable low-rank reconstruction.  Backtracking makes every
    accepted iteration non-increasing in negative log likelihood.
    """

    if max_iter < 1 or learning_rate <= 0 or tolerance < 0:
        raise ValueError("Invalid optimizer settings")
    first = next(iter(data.counts.values()))
    xp = array_namespace(first if initial is None else initial)
    dim = 2**data.n_qubits
    if initial is None:
        initial = project_density_matrix(linear_inversion_pauli(data))
    if initial.shape != (dim, dim):
        raise ValueError("initial density matrix has the wrong shape")
    factor = _initial_factor(initial, xp, rank=rank)
    rho = _factor_to_density(factor, xp)
    objective = negative_log_likelihood(rho, data, epsilon=epsilon)
    history = [objective]

    for _ in range(max_iter):
        state_gradient = _likelihood_gradient(rho, data, xp, epsilon)
        centered = state_gradient - xp.real(xp.trace(state_gradient @ rho)) * eye(
            dim, xp, dtype=complex_dtype(xp), device=device_of(rho)
        )
        factor_gradient = 2.0 * factor @ centered
        step = learning_rate
        accepted = False
        for _ in range(20):
            candidate_factor = factor - step * factor_gradient
            norm = xp.sqrt(xp.real(xp.trace(adjoint(candidate_factor, xp) @ candidate_factor)))
            candidate_factor = candidate_factor / xp.maximum(norm, xp.asarray(epsilon, dtype=norm.dtype))
            candidate = _factor_to_density(candidate_factor, xp)
            candidate_objective = negative_log_likelihood(candidate, data, epsilon=epsilon)
            if candidate_objective <= objective:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        improvement = objective - candidate_objective
        factor, rho, objective = candidate_factor, candidate, candidate_objective
        history.append(objective)
        if improvement <= tolerance * max(1.0, abs(objective)):
            break
    return (rho, history) if return_history else rho
