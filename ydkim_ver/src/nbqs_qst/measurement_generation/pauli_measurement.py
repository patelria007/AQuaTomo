'''
Simulate measuring an n-qubit state in the Pauli bases X, Y, Z (for QST).

The 4^n Pauli strings (identities included) span all 2^n x 2^n matrices, so
rho = (1/2^n) * sum_P <P> P with <P> = Tr(P rho).

Noise is finite sampling only: every shot is an i.i.d. draw from the exact
Born distribution p = diag(U rho U^dagger) -- no Gaussian layer on top.

Hardware-agnostic and pure: the namespace is taken from `rho` via
array_namespace(...), so the same code runs on NumPy/CuPy/JAX/PyTorch
arrays (JAX: set jax_enable_x64 = True first). Randomness comes only from
stdlib random.Random, so a fixed seed reproduces identical outcomes on
every backend. See the adjacent ``pauli_measurement.md`` for the theory.

AI disclosure: this module was generated and revised with OpenAI Codex
assistance. It must not be marked verified until independently reviewed.
'''

import itertools
import operator
import random
from dataclasses import dataclass

from array_api_compat import array_namespace

__all__ = [
    "MeasurementDataset",
    "expectations_from_dataset",
    "generate_measurement_dataset",
    "pauli_expectations",
    "pauli_matrices",
    "pauli_strings",
    "sample_outcomes",
]

# One-qubit Pauli matrices.
_PAULI = {
    "I": ((1, 0), (0, 1)),
    "X": ((0, 1), (1, 0)),
    "Y": ((0, -1j), (1j, 0)),
    "Z": ((1, 0), (0, -1)),
}

# Per-qubit basis change: row k is the eigenbra for outcome k (eigenvalue
# +1, -1), so diag(U rho U^dagger) is the Born distribution. Z is the
# identity: the computational basis already diagonalizes it.
_ROTATION = {
    "Z": ((1, 0), (0, 1)),
    "X": ((2**-0.5, 2**-0.5), (2**-0.5, -(2**-0.5))),
    "Y": ((2**-0.5, -1j * 2**-0.5), (2**-0.5, 1j * 2**-0.5)),
}

# The four one-qubit Paulis stacked, in pauli_strings' lexicographic order.
_PAULI_SET = tuple(_PAULI[c] for c in "IXYZ")


@dataclass(frozen=True)
class MeasurementDataset:
    """Raw and counted outcomes for one complete Pauli measurement run.

    ``settings[i]``, ``outcomes[i]``, and ``counts[i]`` describe the same
    setting. Arrays stay on the backend/device of the input density matrix.
    Outcome indices use qubit 0 as the most significant bit.
    """

    settings: tuple
    outcomes: tuple
    counts: tuple
    shots_per_setting: int
    num_qubits: int
    seed: object = None


def pauli_strings(n):
    """All 4^n Pauli strings of length n, e.g. ['II', 'IX', ..., 'ZZ'].

    n : number of qubits.
    """
    n = _validate_num_qubits(n)
    return ["".join(s) for s in itertools.product("IXYZ", repeat=n)]


def pauli_matrices(n, xp, *, device=None):
    """All Paulis as one (4^n, 2^n, 2^n) array, in pauli_strings order.

    n  : number of qubits.
    xp : array namespace (e.g. from array_namespace(...)).
    device : optional target device for the returned array.
    """
    n = _validate_num_qubits(n)
    single = xp.asarray(
        _PAULI_SET, dtype=xp.complex128, device=device
    )                                                         # (4, 2, 2)
    mats = single
    for _ in range(n - 1):
        m, d, _ = mats.shape                                   # 4^k, 2^k, 2^k
        # Broadcasted outer product P_a (x) P_s': axes (letter, letter,
        # row, row, col, col) flattened in Kronecker order (row = i*2^k + p)
        # to (4^(k+1), 2^(k+1), 2^(k+1)).
        mats = xp.reshape(
            single[:, None, :, None, :, None] * mats[None, :, None, :, None, :],
            (4 * m, 2 * d, 2 * d),
        )
    return mats


def sample_outcomes(rho, basis, shots, *, seed=None):
    """Draw `shots` i.i.d. single-shot outcomes of one measurement setting.

    rho   : (2^n, 2^n) density matrix.
    basis : length-n string over {X, Y, Z}; per-qubit basis, e.g. "XYZ".
    shots : number of independent draws.
    seed  : None -> fresh stdlib stream; int -> random.Random(seed), the
            same stream on every backend.

    Each shot is a categorical draw from the exact Born distribution
    p = diag(U rho U^dagger). Returns int outcome indices in [0, 2^n);
    qubit 0 (basis[0]) is the most significant bit.
    """
    n = _validate_rho_shape(rho)
    basis = _validate_basis(basis, n)
    shots = _validate_shots(shots)
    xp = array_namespace(rho)
    _validate_rho_physicality(rho, xp)
    return _setting_outcomes(rho, basis, shots, random.Random(seed), xp)


def generate_measurement_dataset(rho, shots, *, seed=None):
    """Measure every setting in ``{X,Y,Z}^n`` and retain all observations.

    Returns a :class:`MeasurementDataset` containing backend-native raw
    outcomes and outcome-count vectors. One stdlib ``random.Random`` stream
    is shared across settings, making a fixed seed reproducible across array
    backends.
    """
    n = _validate_rho_shape(rho)
    shots = _validate_shots(shots)
    xp = array_namespace(rho)
    _validate_rho_physicality(rho, xp)
    settings = _measurement_settings(n)
    rng = random.Random(seed)
    outcomes = tuple(
        _setting_outcomes(rho, basis, shots, rng, xp) for basis in settings
    )
    counts = tuple(_outcome_counts(result, 2**n, xp) for result in outcomes)
    return MeasurementDataset(settings, outcomes, counts, shots, n, seed)


def expectations_from_dataset(dataset):
    """Estimate every Pauli expectation using all compatible settings.

    A Pauli containing ``k`` identity letters is compatible with ``3^k``
    settings. Their parity sums and sample counts are accumulated before the
    mean is taken, so no valid observations are discarded or overwritten.
    """
    if not isinstance(dataset, MeasurementDataset):
        raise TypeError("dataset must be a MeasurementDataset")
    expected_settings = 3**dataset.num_qubits
    lengths_match = (
        len(dataset.settings)
        == len(dataset.outcomes)
        == len(dataset.counts)
        == expected_settings
    )
    if not lengths_match:
        raise ValueError(
            "dataset does not contain one entry for every Pauli setting"
        )

    return _expectations_from_runs(
        dataset.settings,
        dataset.outcomes,
        dataset.shots_per_setting,
        dataset.num_qubits,
    )


def pauli_expectations(rho, shots=None, *, seed=None):
    """Return {pauli_string: <P>} for all 4^n Pauli strings.

    rho   : (2^n, 2^n) density matrix.
    shots : None -> exact <P> = Tr(P rho); N -> estimated from N shots.
    seed  : None -> fresh stdlib stream; int -> ONE random.Random(seed)
            streaming across all settings (same on every backend).

    With shots=N, each of the 3^n settings gets N shots. A Pauli with k
    identity letters is compatible with 3^k settings, and all 3^k*N parity
    observations are pooled into its estimate. 'I'*n is trivial and exactly 1.
    """
    n = _validate_rho_shape(rho)
    xp = array_namespace(rho)
    _validate_rho_physicality(rho, xp)

    if shots is None:
        ps = pauli_matrices(n, xp, device=getattr(rho, "device", None))
        # <P> = Tr(P rho) = sum_ij P_ij (rho^T)_ij: elementwise product with
        # rho TRANSPOSED (Y is skew-symmetric, Y^T = -Y, so a direct product
        # would flip signs).
        exact = xp.real(xp.sum(ps * xp.matrix_transpose(rho), axis=(1, 2)))
        # One host transfer for all 4^n values (element-by-element would
        # sync per value on GPU).
        return dict(zip(pauli_strings(n), exact.tolist()))

    shots = _validate_shots(shots)
    settings = _measurement_settings(n)
    rng = random.Random(seed)
    outcomes = (
        _setting_outcomes(rho, basis, shots, rng, xp) for basis in settings
    )
    return _expectations_from_runs(settings, outcomes, shots, n)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _measurement_settings(n):
    """All complete Pauli settings in canonical lexicographic order."""
    return tuple(
        "".join(letters) for letters in itertools.product("XYZ", repeat=n)
    )


def _expectations_from_runs(settings, outcomes, shots, n):
    """Pool compatible parity sums from an iterable of measurement runs."""
    totals = {}
    sample_counts = {}
    for basis, result in zip(settings, outcomes):
        xp = array_namespace(result)
        for pauli, parity_sum in _compatible_expectation_sums(
            result, basis, xp
        ).items():
            totals[pauli] = totals.get(pauli, 0) + parity_sum
            sample_counts[pauli] = sample_counts.get(pauli, 0) + shots

    identity = "I" * n
    estimates = {identity: 1.0}
    estimates.update(
        {
            pauli: float(total / sample_counts[pauli])
            for pauli, total in totals.items()
        }
    )
    return {pauli: estimates[pauli] for pauli in pauli_strings(n)}


def _basis_rotation(basis, xp, *, device=None):
    """Kronecker product of eigenbra basis-change matrices for ``basis``."""
    u = xp.asarray(
        _ROTATION[basis[0]], dtype=xp.complex128, device=device
    )
    for b in basis[1:]:
        factor = xp.asarray(_ROTATION[b], dtype=xp.complex128, device=device)
        u = xp.kron(u, factor)
    return u


def _compatible_expectation_sums(outcomes, basis, xp):
    """Return parity sums for every nontrivial compatible Pauli.

    outcomes : int array of shot outcomes (from _setting_outcomes).
    basis    : length-n {X,Y,Z} setting string.
    xp       : array namespace.

    Compatible = each letter is the setting's letter there or I (the
    setting diagonalizes all of them at once); <P> is the sample mean of
    the +/-1 parities of the outcome bits on P's support.
    """
    n = len(basis)
    # Per-qubit +/-1 eigenvalue of every shot: outcome bit 0 -> +1, 1 -> -1.
    eigs = [1 - 2 * ((outcomes >> (n - 1 - q)) & 1) for q in range(n)]
    sums = {}
    for support in itertools.product([0, 1], repeat=n):
        if not any(support):
            continue
        parity = None
        for q, used in enumerate(support):
            if used:
                parity = eigs[q] if parity is None else parity * eigs[q]
        pauli = "".join(b if used else "I" for b, used in zip(basis, support))
        sums[pauli] = xp.sum(parity)
    return sums


def _setting_outcomes(rho, basis, shots, rng, xp):
    """sample_outcomes with an explicit generator, so pauli_expectations
    can stream ONE rng across all its settings.

    rho   : (2^n, 2^n) density matrix.
    basis : length-n {X,Y,Z} setting string.
    shots : number of draws.
    rng   : stdlib random.Random generator.
    xp    : array namespace.
    """
    device = getattr(rho, "device", None)
    u = _basis_rotation(basis, xp, device=device)
    probs = xp.real(xp.diag(u @ rho @ xp.matrix_transpose(u.conj())))
    # Remove tiny negative round-off and restore unit normalization before
    # inverse-CDF sampling. Valid density matrices are unchanged analytically.
    probs = xp.where(probs > 0, probs, xp.zeros_like(probs))
    probs = probs / xp.sum(probs)
    return _sample_categorical(probs, shots, xp, rng)


def _outcome_counts(outcomes, num_outcomes, xp):
    """Count each outcome without using a backend-specific bincount."""
    labels = xp.arange(
        num_outcomes,
        dtype=xp.int64,
        device=getattr(outcomes, "device", None),
    )
    return xp.sum(outcomes[:, None] == labels[None, :], axis=0)


def _sample_categorical(probs, shots, xp, rng):
    """Draw `shots` i.i.d. outcome indices from `probs` by inverse-CDF
    sampling (one uniform per shot); no count vector or Gaussian layer.

    probs : exact outcome distribution.
    shots : number of draws.
    xp    : array namespace.
    rng   : stdlib random.Random generator.
    """
    u = _uniforms(
        shots, xp, rng, device=getattr(probs, "device", None)
    )
    xq = array_namespace(u)
    cdf = xq.cumsum(probs)
    outcomes = xq.sum(
        (u[:, None] > cdf[None, :]).astype(xq.int64), axis=1
    )
    # A final CDF value infinitesimally below one must not create index d.
    last_outcome = xq.full_like(outcomes, probs.shape[0] - 1)
    return xq.where(outcomes < probs.shape[0], outcomes, last_outcome)


def _uniforms(shots, xp, rng, *, device=None):
    """`shots` uniforms in [0, 1) from the stdlib `rng`, as a backend array.

    shots : number of draws.
    xp    : array namespace.
    rng   : stdlib random.Random generator.

    random.Random is the module's ONLY random source -- one code path,
    identical on every backend. Conversion uses the detected namespace
    directly.
    """
    u = [rng.random() for _ in range(shots)]
    return xp.asarray(u, dtype=xp.float64, device=device)


def _validate_num_qubits(n):
    """Return ``n`` as an int, requiring at least one qubit."""
    if isinstance(n, bool):
        raise TypeError("number of qubits must be an integer")
    try:
        n = operator.index(n)
    except TypeError as exc:
        raise TypeError("number of qubits must be an integer") from exc
    if n < 1:
        raise ValueError("number of qubits must be at least one")
    return n


def _validate_rho_shape(rho):
    """Validate the matrix dimensions and return its qubit count."""
    if getattr(rho, "ndim", None) != 2:
        raise ValueError("rho must be a two-dimensional square matrix")
    rows, columns = rho.shape
    if rows != columns or rows < 2 or rows & (rows - 1):
        raise ValueError("rho shape must be (2**n, 2**n) for n >= 1")
    return rows.bit_length() - 1


def _validate_rho_physicality(rho, xp):
    """Require a finite Hermitian positive semidefinite trace-one state.

    Sampling an invalid matrix after clipping its diagonal probabilities can
    silently fabricate a different distribution.  The eigenspectrum check is
    therefore performed once per public call, before any setting is sampled.
    """
    if not bool(xp.all(xp.isfinite(rho)).tolist()):
        raise ValueError("rho must contain only finite values")

    trace = complex(xp.sum(xp.diag(rho)).tolist())
    if abs(trace.real - 1.0) > 1e-10 or abs(trace.imag) > 1e-10:
        raise ValueError("rho must have real trace one")

    dagger = xp.matrix_transpose(rho.conj())
    hermiticity_error = float(xp.max(xp.abs(rho - dagger)).tolist())
    if hermiticity_error > 1e-10:
        raise ValueError("rho must be Hermitian")

    eigenvalues = xp.real(xp.linalg.eigvalsh(0.5 * (rho + dagger)))
    minimum_eigenvalue = float(eigenvalues[0].tolist())
    if minimum_eigenvalue < -1e-10:
        raise ValueError("rho must be positive semidefinite")


def _validate_basis(basis, n):
    """Validate and return a Pauli measurement setting."""
    if not isinstance(basis, str):
        raise TypeError("basis must be a string")
    if len(basis) != n or any(letter not in "XYZ" for letter in basis):
        raise ValueError(f"basis must contain exactly {n} letters from X, Y, Z")
    return basis


def _validate_shots(shots):
    """Validate and return a strictly positive shot count."""
    if isinstance(shots, bool):
        raise TypeError("shots must be an integer")
    try:
        shots = operator.index(shots)
    except TypeError as exc:
        raise TypeError("shots must be an integer") from exc
    if shots < 1:
        raise ValueError("shots must be at least one")
    return shots
