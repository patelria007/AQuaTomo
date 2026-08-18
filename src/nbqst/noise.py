"""Physical state-noise channels.

The functions here deliberately separate state-preparation/evolution noise
from classical readout noise (implemented in :mod:`nbqst.measurements`).  All
channels are completely positive and trace preserving for their documented
parameter ranges.
"""

from __future__ import annotations

from .backend import adjoint, array_namespace, asarray, complex_dtype, device_of, eye, kron_all
from .operators import pauli


def _validate_density_shape(rho):
    dim = rho.shape[-1]
    n_qubits = dim.bit_length() - 1
    if rho.shape != (dim, dim) or 2**n_qubits != dim:
        raise ValueError("rho must be a square 2^n by 2^n matrix")
    return n_qubits


def _targets(n_qubits: int, targets):
    selected = tuple(range(n_qubits)) if targets is None else tuple(int(value) for value in targets)
    if len(set(selected)) != len(selected) or any(value < 0 or value >= n_qubits for value in selected):
        raise ValueError("targets must contain unique qubit indices")
    return selected


def _apply_local_kraus(rho, single_qubit_kraus, targets=None):
    """Apply one single-qubit Kraus map independently to selected qubits."""

    xp = array_namespace(rho)
    n_qubits = _validate_density_shape(rho)
    identity_2 = pauli("I", xp)
    out = rho
    for target in _targets(n_qubits, targets):
        transformed = xp.zeros_like(out)
        for local_operator in single_qubit_kraus:
            factors = [identity_2] * n_qubits
            factors[target] = local_operator
            operator = kron_all(factors, xp)
            transformed = transformed + operator @ out @ adjoint(operator, xp)
        out = transformed
    return out


def global_depolarizing_channel(rho, probability: float):
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    xp = array_namespace(rho)
    dim = rho.shape[-1]
    identity = eye(dim, xp, dtype=complex_dtype(xp), device=device_of(rho))
    return (1.0 - probability) * rho + probability * identity / dim


def local_depolarizing_channel(rho, probability: float):
    """Apply the same single-qubit depolarizing channel sequentially."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    xp = array_namespace(rho)
    n_qubits = _validate_density_shape(rho)
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


def amplitude_damping_channel(rho, probability: float, *, targets=None):
    """Independent amplitude damping, a first-order model for finite ``T1``.

    ``probability`` is the excited-state decay probability during the modeled
    interval.  With the computational convention used here, ``|1>`` decays to
    ``|0>``.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    xp = array_namespace(rho)
    dtype = complex_dtype(xp)
    device = device_of(rho)
    survival = (1.0 - probability) ** 0.5
    decay = probability**0.5
    operators = (
        asarray([[1.0, 0.0], [0.0, survival]], xp, dtype=dtype, device=device),
        asarray([[0.0, decay], [0.0, 0.0]], xp, dtype=dtype, device=device),
    )
    return _apply_local_kraus(rho, operators, targets=targets)


def phase_damping_channel(rho, probability: float, *, targets=None):
    """Independent phase damping with coherence multiplier ``1-probability``."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    xp = array_namespace(rho)
    dtype = complex_dtype(xp)
    device = device_of(rho)
    keep = (1.0 - probability) ** 0.5
    dephase = probability**0.5
    operators = (
        asarray([[keep, 0.0], [0.0, keep]], xp, dtype=dtype, device=device),
        asarray([[dephase, 0.0], [0.0, 0.0]], xp, dtype=dtype, device=device),
        asarray([[0.0, 0.0], [0.0, dephase]], xp, dtype=dtype, device=device),
    )
    return _apply_local_kraus(rho, operators, targets=targets)


def asymmetric_pauli_channel(rho, *, p_x: float = 0.0, p_y: float = 0.0, p_z: float = 0.0, targets=None):
    """Independent biased Pauli channel for anisotropic stochastic errors."""

    probabilities = (float(p_x), float(p_y), float(p_z))
    if any(value < 0.0 for value in probabilities) or sum(probabilities) > 1.0:
        raise ValueError("p_x, p_y, and p_z must be nonnegative and sum to at most one")
    xp = array_namespace(rho)
    weights = (1.0 - sum(probabilities), *probabilities)
    operators = tuple(weight**0.5 * pauli(label, xp) for weight, label in zip(weights, "IXYZ"))
    return _apply_local_kraus(rho, operators, targets=targets)


def coherent_rotation_channel(rho, angle: float, *, axis: str = "Z", targets=None):
    """Apply a coherent over/under-rotation to selected qubits.

    This unitary channel preserves eigenvalues and purity, making it useful for
    distinguishing coherent calibration error from irreversible decoherence.
    """

    axis = axis.upper()
    if axis not in "XYZ":
        raise ValueError("axis must be X, Y, or Z")
    xp = array_namespace(rho)
    identity_2 = pauli("I", xp)
    rotation = xp.cos(xp.asarray(angle / 2.0)) * identity_2 - 1.0j * xp.sin(
        xp.asarray(angle / 2.0)
    ) * pauli(axis, xp)
    return _apply_local_kraus(rho, (rotation,), targets=targets)
