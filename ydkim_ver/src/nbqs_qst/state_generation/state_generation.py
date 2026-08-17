"""Hardware-agnostic generators for representative n-qubit states.

The public generators cover the state families required by the QST challenge:
local-Haar product states, global Haar pure states, induced Ginibre/Wishart
mixed states, and mixed states with exactly controlled purity.  GHZ and W
states provide deterministic structured references.

Every generator receives a backend ``like`` array, detects its namespace with
``array_namespace(like)``, and keeps returned arrays on the same device.  The
only random source is one stdlib ``random.Random`` stream, so a fixed seed uses
the same random draws on NumPy, CuPy, JAX, and PyTorch.  JAX users must enable
``jax_enable_x64`` before creating ``like``.

AI disclosure: this module was generated with OpenAI Codex assistance on
2026-08-17.  It must not be marked verified until independently reviewed and
validated.  See the adjacent ``state_generation.md`` for derivations, API
semantics, limitations, and references.
"""

import math
import operator
import random
from dataclasses import dataclass

from array_api_compat import array_namespace

__all__ = [
    "GeneratedState",
    "ghz_state",
    "pure_state_overlap",
    "quantum_state_fidelity",
    "random_haar_state",
    "random_mixed_state",
    "random_product_state",
    "random_state_with_purity",
    "state_purity",
    "w_state",
]


@dataclass(frozen=True)
class GeneratedState:
    """A generated target state and the information needed to reproduce it.

    ``rho`` is always present and can be passed directly to the measurement
    simulator.  ``ket`` is present exactly when the generated state is pure.
    ``parameters`` is an immutable tuple of key/value pairs so the frozen
    record does not conceal a mutable dictionary.
    """

    rho: object
    ket: object
    family: str
    num_qubits: int
    seed: object = None
    parameters: tuple = ()

    @property
    def dimension(self):
        """Hilbert-space dimension, equal to ``2**num_qubits``."""
        return 2**self.num_qubits

    @property
    def purity(self):
        """Return ``Tr(rho**2)`` as a Python float."""
        return state_purity(self.rho)

    def metadata(self):
        """Return a serializable description for experiment records."""
        return {
            "family": self.family,
            "num_qubits": self.num_qubits,
            "dimension": self.dimension,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "purity": self.purity,
        }


def random_product_state(like, n, *, seed=None):
    """Generate an unentangled pure state with local-Haar qubit factors.

    Each qubit is an independent normalized two-component complex Gaussian
    vector, which is uniform on the Bloch sphere.  Tensor factors are ordered
    from qubit 0 (most significant bit) to qubit ``n-1``.
    """
    n = _validate_num_qubits(n)
    xp = array_namespace(like)
    device = _device_of(like)
    rng = random.Random(seed)
    ket = _random_product_ket(n, xp, device, rng)
    rho = _density_from_ket(ket, xp)
    return GeneratedState(
        rho=rho,
        ket=ket,
        family="product_haar",
        num_qubits=n,
        seed=seed,
        parameters=(("local_measure", "haar"),),
    )


def random_haar_state(like, n, *, seed=None):
    """Generate a global Haar-random pure state in dimension ``2**n``.

    A length-``2**n`` vector of i.i.d. standard complex Gaussians is
    normalized.  This is equivalent to taking one column of a Haar-random
    unitary, without constructing the full unitary.
    """
    n = _validate_num_qubits(n)
    xp = array_namespace(like)
    device = _device_of(like)
    rng = random.Random(seed)
    ket = _random_haar_ket(2**n, xp, device, rng)
    rho = _density_from_ket(ket, xp)
    return GeneratedState(
        rho=rho,
        ket=ket,
        family="haar_pure",
        num_qubits=n,
        seed=seed,
        parameters=(("measure", "haar"),),
    )


def random_mixed_state(like, n, k=None, *, seed=None):
    """Generate an induced Ginibre/Wishart random density matrix.

    ``G`` is a ``(d, k)`` matrix of i.i.d. complex Gaussian entries and
    ``rho = G G^dagger / Tr(G G^dagger)``, where ``d = 2**n``.  If ``k`` is
    omitted, ``k=d`` gives the Hilbert--Schmidt ensemble.  The state has rank
    ``min(d, k)`` almost surely and ensemble-mean purity
    ``(d + k) / (d*k + 1)``.  ``k=1`` is allowed and reproduces the Haar-pure
    density-matrix ensemble; use ``k>1`` when a strictly mixed target is
    required.
    """
    n = _validate_num_qubits(n)
    d = 2**n
    k = d if k is None else _validate_positive_integer(k, "k")
    xp = array_namespace(like)
    device = _device_of(like)
    rng = random.Random(seed)
    g = _complex_gaussian_array(d * k, xp, device, rng)
    g = xp.reshape(g, (d, k))
    gram = g @ xp.matrix_transpose(g.conj())
    trace = xp.real(xp.sum(xp.diag(gram)))
    rho = gram / trace
    ket = g[:, 0] / xp.sqrt(trace) if k == 1 else None
    return GeneratedState(
        rho=rho,
        ket=ket,
        family="induced_mixed",
        num_qubits=n,
        seed=seed,
        parameters=(
            ("environment_dimension", k),
            ("expected_rank", min(d, k)),
            ("mean_purity", (d + k) / (d * k + 1)),
        ),
    )


def random_state_with_purity(
    like,
    n,
    target_purity,
    *,
    seed=None,
    base="haar",
):
    """Generate a random state with exactly prescribed analytical purity.

    Starting from a Haar or product pure state ``rho0``, return

    ``rho = alpha*rho0 + (1-alpha)*I/d``

    with ``alpha = sqrt((gamma - 1/d)/(1 - 1/d))``.  Thus the requested
    ``target_purity=gamma`` can be any value in ``[1/d, 1]``.  This is a
    controlled benchmark family, not a uniform fixed-purity ensemble.
    """
    n = _validate_num_qubits(n)
    d = 2**n
    target_purity = _validate_target_purity(target_purity, d)
    if base not in ("haar", "product"):
        raise ValueError("base must be 'haar' or 'product'")

    xp = array_namespace(like)
    device = _device_of(like)
    rng = random.Random(seed)
    if base == "haar":
        source_ket = _random_haar_ket(d, xp, device, rng)
    else:
        source_ket = _random_product_ket(n, xp, device, rng)

    source_rho = _density_from_ket(source_ket, xp)
    alpha = math.sqrt(
        max(0.0, (target_purity - 1.0 / d) / (1.0 - 1.0 / d))
    )
    identity = xp.eye(d, dtype=xp.complex128, device=device)
    rho = alpha * source_rho + (1.0 - alpha) * identity / d
    ket = source_ket if target_purity == 1.0 else None
    return GeneratedState(
        rho=rho,
        ket=ket,
        family="purity_controlled",
        num_qubits=n,
        seed=seed,
        parameters=(
            ("base", base),
            ("target_purity", target_purity),
            ("alpha", alpha),
        ),
    )


def ghz_state(like, n):
    """Return ``(|0...0> + |1...1>)/sqrt(2)`` and its density matrix."""
    n = _validate_num_qubits(n)
    d = 2**n
    xp = array_namespace(like)
    device = _device_of(like)
    scale = 2**-0.5
    values = [scale if index in (0, d - 1) else 0j for index in range(d)]
    ket = xp.asarray(values, dtype=xp.complex128, device=device)
    return GeneratedState(
        rho=_density_from_ket(ket, xp),
        ket=ket,
        family="ghz",
        num_qubits=n,
    )


def w_state(like, n):
    """Return the equal superposition of all one-excitation basis states."""
    n = _validate_num_qubits(n)
    d = 2**n
    xp = array_namespace(like)
    device = _device_of(like)
    excitation_indices = frozenset(2 ** (n - 1 - q) for q in range(n))
    scale = n**-0.5
    values = [scale if index in excitation_indices else 0j for index in range(d)]
    ket = xp.asarray(values, dtype=xp.complex128, device=device)
    return GeneratedState(
        rho=_density_from_ket(ket, xp),
        ket=ket,
        family="w",
        num_qubits=n,
    )


def state_purity(rho):
    """Return ``Tr(rho**2)`` using the backend of ``rho``.

    The elementwise expression uses ``rho`` transposed, since
    ``Tr(rho**2) = sum_ij rho_ij rho_ji``.  The result is synchronized once
    and returned as a Python float for experiment logging.
    """
    _validate_density_shape(rho)
    xp = array_namespace(rho)
    value = xp.real(xp.sum(rho * xp.matrix_transpose(rho)))
    return float(value.tolist())


def pure_state_overlap(psi_true, psi_estimated):
    """Return ``|<psi_true|psi_estimated>|**2`` as a Python float."""
    _validate_matching_kets(psi_true, psi_estimated)
    xp = array_namespace(psi_true, psi_estimated)
    inner = xp.sum(psi_true.conj() * psi_estimated)
    value = xp.real(inner.conj() * inner)
    return float(value.tolist())


def quantum_state_fidelity(rho, sigma):
    """Return squared Uhlmann fidelity between two density matrices.

    This follows the challenge convention
    ``F=(Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2``.  Tiny negative eigenvalues
    caused by Hermitian eigensolver round-off are clipped to zero.
    """
    _validate_matching_density_matrices(rho, sigma)
    xp = array_namespace(rho, sigma)
    eigenvalues, eigenvectors = xp.linalg.eigh(_hermitian_part(rho, xp))
    clipped = _resolved_nonnegative_spectrum(eigenvalues, xp)
    sqrt_rho = (
        eigenvectors
        @ xp.diag(xp.sqrt(clipped))
        @ xp.matrix_transpose(eigenvectors.conj())
    )
    middle = _hermitian_part(sqrt_rho @ sigma @ sqrt_rho, xp)
    middle_eigenvalues = xp.linalg.eigvalsh(middle)
    middle_clipped = _resolved_nonnegative_spectrum(middle_eigenvalues, xp)
    root_fidelity = xp.sum(xp.sqrt(middle_clipped))
    fidelity = xp.real(root_fidelity * root_fidelity)
    return min(1.0, max(0.0, float(fidelity.tolist())))


# ---------------------------------------------------------------------------
# Internal construction helpers
# ---------------------------------------------------------------------------


def _random_product_ket(n, xp, device, rng):
    local_kets = tuple(
        _normalized_complex_gaussian(2, xp, device, rng) for _ in range(n)
    )
    ket = local_kets[0]
    for local in local_kets[1:]:
        ket = xp.kron(ket, local)
    return ket


def _random_haar_ket(d, xp, device, rng):
    return _normalized_complex_gaussian(d, xp, device, rng)


def _normalized_complex_gaussian(length, xp, device, rng):
    values = _complex_gaussian_array(length, xp, device, rng)
    norm_squared = xp.real(xp.sum(values.conj() * values))
    return values / xp.sqrt(norm_squared)


def _complex_gaussian_array(length, xp, device, rng):
    values = [
        complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        for _ in range(length)
    ]
    return xp.asarray(values, dtype=xp.complex128, device=device)


def _density_from_ket(ket, xp):
    return ket[:, None] * ket.conj()[None, :]


def _hermitian_part(matrix, xp):
    return 0.5 * (matrix + xp.matrix_transpose(matrix.conj()))


def _resolved_nonnegative_spectrum(eigenvalues, xp):
    """Clip negative and numerically unresolved positive eigenvalues.

    Square roots amplify round-off eigenvalues of an analytically rank-deficient
    state.  A dimension-scaled eigensolver tolerance prevents a pure state's
    self-fidelity from acquiring an artificial ``sqrt(eps)`` excess.
    """
    scale = xp.max(xp.abs(eigenvalues))
    tolerance = 10 * eigenvalues.shape[0] * xp.finfo(eigenvalues.dtype).eps * scale
    return xp.where(
        eigenvalues > tolerance,
        eigenvalues,
        xp.zeros_like(eigenvalues),
    )


def _device_of(array):
    return getattr(array, "device", None)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_num_qubits(n):
    if isinstance(n, bool):
        raise TypeError("number of qubits must be an integer")
    return _validate_positive_integer(n, "number of qubits")


def _validate_positive_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def _validate_target_purity(value, d):
    if isinstance(value, bool):
        raise TypeError("target_purity must be a real number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("target_purity must be a real number") from exc
    if not math.isfinite(value):
        raise ValueError("target_purity must be finite")
    lower = 1.0 / d
    tolerance = 1e-14
    if value < lower - tolerance or value > 1.0 + tolerance:
        raise ValueError(f"target_purity must be in [{lower}, 1.0]")
    return min(1.0, max(lower, value))


def _validate_density_shape(rho):
    if getattr(rho, "ndim", None) != 2:
        raise ValueError("density matrix must be two-dimensional")
    rows, columns = rho.shape
    if rows != columns or rows < 2 or rows & (rows - 1):
        raise ValueError("density matrix shape must be (2**n, 2**n) for n >= 1")


def _validate_matching_density_matrices(rho, sigma):
    _validate_density_shape(rho)
    _validate_density_shape(sigma)
    if rho.shape != sigma.shape:
        raise ValueError("density matrices must have the same shape")


def _validate_matching_kets(psi_true, psi_estimated):
    for ket in (psi_true, psi_estimated):
        if getattr(ket, "ndim", None) != 1:
            raise ValueError("state vectors must be one-dimensional")
        length = ket.shape[0]
        if length < 2 or length & (length - 1):
            raise ValueError("state-vector length must be 2**n for n >= 1")
    if psi_true.shape != psi_estimated.shape:
        raise ValueError("state vectors must have the same shape")
