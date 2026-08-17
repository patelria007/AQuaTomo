"""Backend-neutral tomography quality metrics."""

from __future__ import annotations

from .backend import adjoint, array_namespace
from .denoise import hermitize


def purity(rho):
    xp = array_namespace(rho)
    return xp.real(xp.trace(rho @ rho))


def hilbert_schmidt_distance(rho, sigma):
    xp = array_namespace(rho, sigma)
    delta = rho - sigma
    return xp.sqrt(xp.maximum(xp.real(xp.trace(adjoint(delta, xp) @ delta)), xp.asarray(0.0)))


def fidelity(rho, sigma):
    """Squared Uhlmann fidelity in [0, 1]."""

    xp = array_namespace(rho, sigma)
    values, vectors = xp.linalg.eigh(hermitize(rho))
    values = xp.maximum(xp.real(values), xp.asarray(0.0, dtype=values.dtype))
    sqrt_rho = (vectors * xp.sqrt(values)[None, :]) @ adjoint(vectors, xp)
    middle = hermitize(sqrt_rho @ sigma @ sqrt_rho)
    inner_values = xp.linalg.eigvalsh(middle)
    trace_root = xp.sum(xp.sqrt(xp.maximum(xp.real(inner_values), xp.asarray(0.0, dtype=inner_values.dtype))))
    return xp.minimum(xp.maximum(trace_root * trace_root, xp.asarray(0.0)), xp.asarray(1.0))


def trace_distance(rho, sigma):
    xp = array_namespace(rho, sigma)
    singular_values = xp.linalg.svd(rho - sigma, full_matrices=False)[1]
    return 0.5 * xp.sum(xp.real(singular_values))


def minimum_eigenvalue(rho):
    xp = array_namespace(rho)
    return xp.min(xp.real(xp.linalg.eigvalsh(hermitize(rho))))

