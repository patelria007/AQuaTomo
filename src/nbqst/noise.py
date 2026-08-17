"""Physical state-noise channels."""

from __future__ import annotations

from .backend import array_namespace, complex_dtype, device_of, kron_all
from .operators import pauli


def global_depolarizing_channel(rho, probability: float):
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    xp = array_namespace(rho)
    dim = rho.shape[-1]
    identity = xp.eye(dim, dtype=complex_dtype(xp), device=device_of(rho))
    return (1.0 - probability) * rho + probability * identity / dim


def local_depolarizing_channel(rho, probability: float):
    """Apply the same single-qubit depolarizing channel sequentially."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    xp = array_namespace(rho)
    dim = rho.shape[-1]
    n_qubits = dim.bit_length() - 1
    if 2**n_qubits != dim or rho.shape[-2] != dim:
        raise ValueError("rho must be a square 2^n by 2^n matrix")
    identity_2 = pauli("I", xp)
    out = rho
    for target in range(n_qubits):
        transformed = xp.zeros_like(out)
        for axis in "XYZ":
            factors = [identity_2] * n_qubits
            factors[target] = pauli(axis, xp)
            operator = kron_all(factors, xp)
            transformed = transformed + operator @ out @ xp.conj(xp.swapaxes(operator, -1, -2))
        out = (1.0 - probability) * out + (probability / 3.0) * transformed
    return out

