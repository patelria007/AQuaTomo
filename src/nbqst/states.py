"""Quantum-state generators with reproducible host RNG and native arrays."""

from __future__ import annotations

import numpy as np

from .backend import adjoint, asarray, complex_dtype


def _rng(rng=None):
    return np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng


def ket_to_density(ket, xp):
    ket = xp.asarray(ket)
    return ket[:, None] * xp.conj(ket)[None, :]


def random_product_state(n_qubits: int, *, xp=np, rng=None):
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    generator = _rng(rng)
    ket = np.array([1.0 + 0.0j])
    for _ in range(n_qubits):
        z = generator.normal(size=2) + 1j * generator.normal(size=2)
        z /= np.linalg.norm(z)
        ket = np.kron(ket, z)
    native = asarray(ket, xp, dtype=complex_dtype(xp))
    return ket_to_density(native, xp)


def haar_random_pure(n_qubits: int, *, xp=np, rng=None):
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    generator = _rng(rng)
    dim = 2**n_qubits
    ket = generator.normal(size=dim) + 1j * generator.normal(size=dim)
    ket /= np.linalg.norm(ket)
    native = asarray(ket, xp, dtype=complex_dtype(xp))
    return ket_to_density(native, xp)


def random_mixed_state(n_qubits: int, *, rank=None, xp=np, rng=None):
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    dim = 2**n_qubits
    rank = dim if rank is None else int(rank)
    if not 1 <= rank <= dim:
        raise ValueError(f"rank must be between 1 and {dim}")
    generator = _rng(rng)
    ginibre = generator.normal(size=(dim, rank)) + 1j * generator.normal(size=(dim, rank))
    native = asarray(ginibre, xp, dtype=complex_dtype(xp))
    rho = native @ adjoint(native, xp)
    return rho / xp.real(xp.trace(rho))


def ghz_state(n_qubits: int, *, xp=np):
    if n_qubits < 2:
        raise ValueError("GHZ state requires at least two qubits")
    dim = 2**n_qubits
    ket = np.zeros(dim, dtype=complex)
    ket[0] = ket[-1] = 2.0**-0.5
    native = asarray(ket, xp, dtype=complex_dtype(xp))
    return ket_to_density(native, xp)

