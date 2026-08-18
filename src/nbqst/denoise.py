"""Training-free, physics-constrained density-matrix denoisers."""

from __future__ import annotations

from .backend import (
    adjoint,
    arange,
    array_namespace,
    asarray,
    complex_dtype,
    cumulative_sum,
    device_of,
    eye,
    real_dtype,
)


def hermitize(matrix):
    xp = array_namespace(matrix)
    return 0.5 * (matrix + adjoint(matrix, xp))


def _simplex_project(eigenvalues, xp):
    """Project real values onto {x >= 0, sum(x) = 1} without mutation."""

    d = eigenvalues.shape[0]
    ordered = xp.sort(eigenvalues)[::-1]
    cssv = cumulative_sum(ordered, xp) - 1.0
    indices = arange(1, d + 1, xp=xp, dtype=ordered.dtype, device=device_of(eigenvalues))
    candidates = cssv / indices
    valid = ordered - candidates > 0
    negative_inf = asarray(float("-inf"), xp, dtype=ordered.dtype, device=device_of(eigenvalues))
    theta = xp.max(xp.where(valid, candidates, negative_inf))
    return xp.maximum(eigenvalues - theta, asarray(0.0, xp, dtype=eigenvalues.dtype, device=device_of(eigenvalues)))


def project_density_matrix(matrix):
    """Nearest PSD, trace-one matrix in Frobenius norm."""

    xp = array_namespace(matrix)
    clean = hermitize(matrix)
    values, vectors = xp.linalg.eigh(clean)
    weights = _simplex_project(xp.real(values), xp)
    projected = (vectors * weights[None, :]) @ adjoint(vectors, xp)
    return hermitize(projected)


def low_rank_projection(matrix, rank: int, *, shrinkage: float = 0.0):
    """Truncate and shrink the spectrum, then enforce a physical state."""

    xp = array_namespace(matrix)
    dim = matrix.shape[-1]
    if not 1 <= rank <= dim:
        raise ValueError(f"rank must be between 1 and {dim}")
    if shrinkage < 0:
        raise ValueError("shrinkage must be nonnegative")
    values, vectors = xp.linalg.eigh(hermitize(matrix))
    positions = arange(dim, xp=xp, device=device_of(values))
    keep = positions >= dim - rank
    zero = asarray(0.0, xp, dtype=values.dtype, device=device_of(values))
    weights = xp.where(keep, xp.maximum(xp.real(values) - shrinkage, zero), zero)
    total = xp.sum(weights)
    fallback = xp.where(keep, xp.asarray(1.0 / rank, dtype=values.dtype), zero)
    safe_total = xp.maximum(total, asarray(1e-15, xp, dtype=values.dtype, device=device_of(values)))
    normalized = xp.where(total > 1e-15, weights / safe_total, fallback)
    rho = (vectors * normalized[None, :]) @ adjoint(vectors, xp)
    return hermitize(rho)


def depolarizing_shrinkage(matrix, alpha: float):
    """Wiener-style shrinkage toward the maximally mixed state."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    xp = array_namespace(matrix)
    dim = matrix.shape[-1]
    identity = eye(dim, xp, dtype=complex_dtype(xp), device=device_of(matrix)) / dim
    return project_density_matrix(alpha * hermitize(matrix) + (1.0 - alpha) * identity)


def select_shrinkage_alpha(matrix, validation_data, candidates=None):
    """Select shrinkage by held-out multinomial negative log likelihood."""

    from .reconstruction import negative_log_likelihood

    candidates = tuple(i / 20 for i in range(21)) if candidates is None else tuple(candidates)
    scored = [(negative_log_likelihood(depolarizing_shrinkage(matrix, a), validation_data), a) for a in candidates]
    return min(scored, key=lambda pair: pair[0])[1]
