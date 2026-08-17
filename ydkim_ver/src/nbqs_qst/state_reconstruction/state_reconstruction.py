"""Hardware-agnostic reconstruction of states from Pauli measurement counts.

The module provides three reconstruction paths:

* linear inversion in the orthogonal Pauli operator basis,
* projected least squares (PLS), and
* physical maximum-likelihood estimation for the exact multinomial count
  model, using projected gradient descent with backtracking.

Core logic detects the array namespace from the backend-native count arrays
and uses no backend random-number generator.  JAX users must enable 64-bit
mode before creating their arrays.  See ``state_reconstruction.md`` for the
derivations, algorithm choices, limitations, and validation contract.

AI disclosure: this code was generated with AI assistance on 2026-08-17 and
must not be marked verified until it has been independently reviewed.
"""

from __future__ import annotations

import itertools
import math
import operator
from dataclasses import dataclass

from array_api_compat import array_namespace

from ..measurement_generation.pauli_measurement import (
    MeasurementDataset,
    pauli_matrices,
    pauli_strings,
)

__all__ = [
    "ReconstructionResult",
    "linear_inversion",
    "maximum_likelihood",
    "negative_log_likelihood",
    "project_density_matrix",
    "projected_least_squares",
    "purity",
    "reconstruct",
    "state_fidelity",
    "trace_distance",
]


_ROTATION = {
    "Z": ((1, 0), (0, 1)),
    "X": ((2**-0.5, 2**-0.5), (2**-0.5, -(2**-0.5))),
    "Y": ((2**-0.5, -1j * 2**-0.5), (2**-0.5, 1j * 2**-0.5)),
}


@dataclass(frozen=True)
class ReconstructionResult:
    """A reconstructed state together with convergence and physics checks."""

    rho: object
    method: str
    converged: bool
    iterations: int
    objective: float
    objective_history: tuple
    trace_error: float
    hermiticity_error: float
    min_eigenvalue: float


def linear_inversion(dataset):
    """Return the Pauli-basis linear-inversion estimate for ``dataset``.

    All compatible settings are pooled directly from integer count vectors.
    Raw per-shot outcomes are not required.  The output is Hermitian and trace
    one up to floating-point precision, but finite-shot noise can make it
    non-positive; no eigenvalue is clipped.
    """
    xp, device, n, dimension, _ = _validate_dataset(dataset)
    expectations = _expectations_from_counts(dataset, xp, n, dimension)
    coefficients = xp.asarray(
        [expectations[label] for label in pauli_strings(n)],
        dtype=xp.float64,
        device=device,
    )
    paulis = pauli_matrices(n, xp, device=device)
    rho = xp.sum(coefficients[:, None, None] * paulis, axis=0) / dimension
    return _hermitize(rho, xp)


def project_density_matrix(matrix):
    """Frobenius-project a square matrix onto the density-matrix set.

    The matrix is first Hermitized.  Its eigenvalues are then projected onto
    the probability simplex while the eigenvectors are retained.
    """
    xp = array_namespace(matrix)
    _validate_square_matrix(matrix, "matrix")
    hermitian = _hermitize(matrix, xp)
    eigenvalues, eigenvectors = xp.linalg.eigh(hermitian)
    real_values = xp.real(eigenvalues)
    threshold = _simplex_threshold(real_values)
    shifted = real_values - threshold
    weights = xp.where(shifted > 0, shifted, xp.zeros_like(shifted))
    weights = weights / xp.sum(weights)
    projected = (
        eigenvectors * weights[None, :]
    ) @ xp.matrix_transpose(eigenvectors.conj())
    return _hermitize(projected, xp)


def projected_least_squares(dataset):
    """Return the physical Frobenius projection of linear inversion."""
    return project_density_matrix(linear_inversion(dataset))


def negative_log_likelihood(
    rho,
    dataset,
    *,
    normalized=True,
    probability_tolerance=1e-14,
):
    """Return multinomial NLL, omitting constants independent of ``rho``.

    By default the count-weighted objective is divided by the total number of
    shots.  This positive rescaling does not alter the MLE.  If an observed
    outcome has probability at or below ``probability_tolerance``, the result
    is positive infinity rather than an silently epsilon-clipped likelihood.
    """
    xp, _, _, dimension, total_shots = _validate_dataset(dataset)
    _validate_state_for_dataset(rho, xp, dimension)
    tolerance = _validate_nonnegative_real(
        probability_tolerance, "probability_tolerance"
    )
    value = _nll_array(rho, dataset, xp, total_shots, tolerance)
    result = math.inf if value is None else _as_float(value)
    return result if normalized else result * total_shots


def maximum_likelihood(
    dataset,
    *,
    initial="mixed",
    max_iterations=500,
    tolerance=1e-9,
    initial_step=1.0,
    backtrack_factor=0.5,
    armijo=1e-4,
    probability_tolerance=1e-14,
):
    """Reconstruct the multinomial MLE with projected gradient descent.

    Each candidate is projected onto the positive, trace-one state space.
    Backtracking accepts only finite candidates satisfying an Armijo decrease
    condition.  The returned history contains the normalized NLL of the
    initial state and every accepted iterate.
    """
    xp, device, _, dimension, total_shots = _validate_dataset(dataset)
    max_iterations = _validate_positive_integer(
        max_iterations, "max_iterations"
    )
    tolerance = _validate_positive_real(tolerance, "tolerance")
    initial_step = _validate_positive_real(initial_step, "initial_step")
    backtrack_factor = _validate_open_unit_interval(
        backtrack_factor, "backtrack_factor"
    )
    armijo = _validate_open_unit_interval(armijo, "armijo")
    probability_tolerance = _validate_nonnegative_real(
        probability_tolerance, "probability_tolerance"
    )

    rho = _initial_state(initial, dataset, xp, device, dimension)
    cost, gradient = _cost_and_gradient(
        rho, dataset, xp, total_shots, probability_tolerance
    )
    if cost is None:
        raise ValueError(
            "initial state assigns zero probability to an observed outcome"
        )

    cost_value = _as_float(cost)
    history = [cost_value]
    converged = False
    accepted_iterations = 0
    step_hint = initial_step
    stalled_iterations = 0

    for _ in range(max_iterations):
        step = step_hint
        accepted = False
        candidate = rho
        candidate_cost = cost
        step_norm = math.inf
        objective_change = 0.0

        for _ in range(40):
            proposal = project_density_matrix(rho - step * gradient)
            delta = proposal - rho
            directional = _as_float(
                xp.real(xp.sum(gradient.conj() * delta))
            )
            proposal_cost = _nll_array(
                proposal,
                dataset,
                xp,
                total_shots,
                probability_tolerance,
            )
            if proposal_cost is not None:
                proposal_value = _as_float(proposal_cost)
                armijo_bound = cost_value + armijo * directional
                if proposal_value <= armijo_bound:
                    candidate = proposal
                    candidate_cost = proposal_cost
                    step_norm = _matrix_norm(delta, xp)
                    objective_change = cost_value - proposal_value
                    accepted = True
                    break
            step *= backtrack_factor

        if not accepted:
            break

        rho = candidate
        cost = candidate_cost
        cost_value = _as_float(cost)
        history.append(cost_value)
        accepted_iterations += 1
        step_hint = min(step / backtrack_factor, initial_step * 32.0)

        small_objective_change = objective_change <= tolerance * (
            1.0 + abs(cost_value)
        )
        stalled_iterations = (
            stalled_iterations + 1 if small_objective_change else 0
        )
        if step_norm <= tolerance or stalled_iterations >= 5:
            converged = True
            break

        cost, gradient = _cost_and_gradient(
            rho, dataset, xp, total_shots, probability_tolerance
        )
        if cost is None:
            raise RuntimeError("accepted MLE iterate has an invalid likelihood")
        cost_value = _as_float(cost)

    return _result(
        rho,
        "maximum_likelihood",
        converged,
        accepted_iterations,
        cost_value,
        tuple(history),
    )


def reconstruct(dataset, method="mle", **kwargs):
    """Run ``linear``, ``pls``, or ``mle`` and return diagnostics uniformly."""
    if method == "mle":
        return maximum_likelihood(dataset, **kwargs)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"{method} reconstruction has no options: {unexpected}")
    if method == "linear":
        rho = linear_inversion(dataset)
    elif method == "pls":
        rho = projected_least_squares(dataset)
    else:
        raise ValueError("method must be 'linear', 'pls', or 'mle'")
    objective = negative_log_likelihood(rho, dataset)
    return _result(rho, method, True, 0, objective, (objective,))


def purity(rho):
    """Return ``Tr(rho^2)`` using the input array namespace."""
    xp = array_namespace(rho)
    _validate_square_matrix(rho, "rho")
    return _as_float(xp.real(xp.sum((rho @ rho) * xp.eye(
        rho.shape[0], dtype=xp.complex128, device=getattr(rho, "device", None)
    ))))


def state_fidelity(rho, sigma):
    """Return squared Uhlmann fidelity between two density matrices."""
    xp = array_namespace(rho, sigma)
    dimension = _validate_square_matrix(rho, "rho")
    if _validate_square_matrix(sigma, "sigma") != dimension:
        raise ValueError("rho and sigma must have the same shape")
    sqrt_rho = _positive_square_root(rho, xp)
    middle = _hermitize(sqrt_rho @ sigma @ sqrt_rho, xp)
    eigenvalues = xp.real(xp.linalg.eigvalsh(middle))
    nonnegative = xp.where(
        eigenvalues > 0, eigenvalues, xp.zeros_like(eigenvalues)
    )
    value = _as_float(xp.sum(xp.sqrt(nonnegative)) ** 2)
    return min(1.0, max(0.0, value))


def trace_distance(rho, sigma):
    """Return ``0.5 * ||rho - sigma||_1`` for Hermitian state matrices."""
    xp = array_namespace(rho, sigma)
    dimension = _validate_square_matrix(rho, "rho")
    if _validate_square_matrix(sigma, "sigma") != dimension:
        raise ValueError("rho and sigma must have the same shape")
    eigenvalues = xp.linalg.eigvalsh(_hermitize(rho - sigma, xp))
    return 0.5 * _as_float(xp.sum(xp.abs(eigenvalues)))


# ---------------------------------------------------------------------------
# Likelihood and linear-algebra internals
# ---------------------------------------------------------------------------


def _cost_and_gradient(rho, dataset, xp, total_shots, probability_tolerance):
    device = getattr(rho, "device", None)
    cost = xp.asarray(0.0, dtype=xp.float64, device=device)
    gradient = xp.zeros_like(rho)
    for basis, counts in zip(dataset.settings, dataset.counts):
        rotation = _basis_rotation(basis, xp, device=device)
        dagger = xp.matrix_transpose(rotation.conj())
        probabilities = xp.real(xp.diag(rotation @ rho @ dagger))
        observed = counts > 0
        invalid = xp.any(observed & (probabilities <= probability_tolerance))
        if bool(invalid.tolist()):
            return None, None
        safe = xp.where(observed, probabilities, xp.ones_like(probabilities))
        frequencies = counts / total_shots
        cost = cost - xp.sum(frequencies * xp.log(safe))
        weights = xp.where(
            observed, frequencies / safe, xp.zeros_like(probabilities)
        )
        gradient = gradient - dagger @ xp.diag(weights) @ rotation
    return cost, _hermitize(gradient, xp)


def _nll_array(rho, dataset, xp, total_shots, probability_tolerance):
    device = getattr(rho, "device", None)
    cost = xp.asarray(0.0, dtype=xp.float64, device=device)
    for basis, counts in zip(dataset.settings, dataset.counts):
        rotation = _basis_rotation(basis, xp, device=device)
        dagger = xp.matrix_transpose(rotation.conj())
        probabilities = xp.real(xp.diag(rotation @ rho @ dagger))
        observed = counts > 0
        invalid = xp.any(observed & (probabilities <= probability_tolerance))
        if bool(invalid.tolist()):
            return None
        safe = xp.where(observed, probabilities, xp.ones_like(probabilities))
        cost = cost - xp.sum((counts / total_shots) * xp.log(safe))
    return cost


def _basis_rotation(basis, xp, *, device=None):
    rotation = xp.asarray(
        _ROTATION[basis[0]], dtype=xp.complex128, device=device
    )
    for letter in basis[1:]:
        factor = xp.asarray(
            _ROTATION[letter], dtype=xp.complex128, device=device
        )
        rotation = xp.kron(rotation, factor)
    return rotation


def _expectations_from_counts(dataset, xp, n, dimension):
    """Pool compatible Pauli parities using count vectors only."""
    device = getattr(dataset.counts[0], "device", None)
    totals = {}
    sample_counts = {}
    for basis, counts in zip(dataset.settings, dataset.counts):
        for support in itertools.product((0, 1), repeat=n):
            if not any(support):
                continue
            pauli = "".join(
                letter if used else "I"
                for letter, used in zip(basis, support)
            )
            signs = []
            for outcome in range(dimension):
                parity = 1
                for qubit, used in enumerate(support):
                    if used:
                        bit = (outcome >> (n - 1 - qubit)) & 1
                        parity *= 1 - 2 * bit
                signs.append(parity)
            eigenvalues = xp.asarray(
                signs, dtype=xp.int64, device=device
            )
            parity_sum = xp.sum(counts * eigenvalues)
            totals[pauli] = totals.get(pauli, 0) + parity_sum
            sample_counts[pauli] = (
                sample_counts.get(pauli, 0) + dataset.shots_per_setting
            )

    identity = "I" * n
    estimates = {identity: 1.0}
    estimates.update(
        {
            label: _as_float(total / sample_counts[label])
            for label, total in totals.items()
        }
    )
    return estimates


def _positive_square_root(matrix, xp):
    hermitian = _hermitize(matrix, xp)
    values, vectors = xp.linalg.eigh(hermitian)
    values = xp.real(values)
    values = xp.where(values > 0, values, xp.zeros_like(values))
    return (vectors * xp.sqrt(values)[None, :]) @ xp.matrix_transpose(
        vectors.conj()
    )


def _simplex_threshold(values):
    descending = sorted(
        (float(value) for value in values.tolist()), reverse=True
    )
    cumulative = 0.0
    active = 1
    for index, value in enumerate(descending, start=1):
        cumulative += value
        candidate = (cumulative - 1.0) / index
        if value > candidate:
            active = index
    return (sum(descending[:active]) - 1.0) / active


def _hermitize(matrix, xp):
    return 0.5 * (matrix + xp.matrix_transpose(matrix.conj()))


def _matrix_norm(matrix, xp):
    return _as_float(xp.sqrt(xp.sum(xp.abs(matrix) ** 2)))


def _result(rho, method, converged, iterations, objective, history):
    xp = array_namespace(rho)
    trace = _as_complex(xp.sum(xp.diag(rho)))
    trace_error = abs(trace - 1.0)
    hermiticity_error = _matrix_norm(
        rho - xp.matrix_transpose(rho.conj()), xp
    )
    minimum = _as_float(xp.real(xp.linalg.eigvalsh(_hermitize(rho, xp))[0]))
    return ReconstructionResult(
        rho=rho,
        method=method,
        converged=converged,
        iterations=iterations,
        objective=objective,
        objective_history=history,
        trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        min_eigenvalue=minimum,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_dataset(dataset):
    if not isinstance(dataset, MeasurementDataset):
        raise TypeError("dataset must be a MeasurementDataset")
    n = _validate_positive_integer(dataset.num_qubits, "dataset.num_qubits")
    shots = _validate_positive_integer(
        dataset.shots_per_setting, "dataset.shots_per_setting"
    )
    settings = tuple(
        "".join(letters) for letters in itertools.product("XYZ", repeat=n)
    )
    if tuple(dataset.settings) != settings:
        raise ValueError("dataset settings must be the complete lexicographic XYZ set")
    if len(dataset.counts) != len(settings):
        raise ValueError("dataset must contain one count vector per setting")
    if not dataset.counts:
        raise ValueError("dataset counts cannot be empty")

    xp = array_namespace(dataset.counts[0])
    device = getattr(dataset.counts[0], "device", None)
    dimension = 2**n
    for counts in dataset.counts:
        if getattr(counts, "ndim", None) != 1 or counts.shape != (dimension,):
            raise ValueError("each count vector must have shape (2**num_qubits,)")
        values = counts.tolist()
        checked = []
        for value in values:
            if isinstance(value, bool):
                raise TypeError("counts must contain integers")
            try:
                integer = operator.index(value)
            except TypeError as exc:
                raise TypeError("counts must contain integers") from exc
            if integer < 0:
                raise ValueError("counts must be nonnegative")
            checked.append(integer)
        if sum(checked) != shots:
            raise ValueError("each count vector must sum to shots_per_setting")
    return xp, device, n, dimension, shots * len(settings)


def _validate_state_for_dataset(rho, xp, dimension):
    if array_namespace(rho) is not xp:
        raise TypeError("rho and dataset counts must use the same array namespace")
    if _validate_square_matrix(rho, "rho") != dimension:
        raise ValueError("rho shape does not match dataset.num_qubits")


def _validate_square_matrix(matrix, name):
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError(f"{name} must be a two-dimensional square matrix")
    rows, columns = matrix.shape
    if rows != columns or rows < 1:
        raise ValueError(f"{name} must be a nonempty square matrix")
    return rows


def _initial_state(initial, dataset, xp, device, dimension):
    if isinstance(initial, str):
        if initial == "mixed":
            return xp.eye(
                dimension, dtype=xp.complex128, device=device
            ) / dimension
        if initial in ("linear", "pls"):
            return projected_least_squares(dataset)
        raise ValueError("initial must be 'mixed', 'linear', 'pls', or an array")
    state = xp.asarray(initial, dtype=xp.complex128, device=device)
    if _validate_square_matrix(state, "initial") != dimension:
        raise ValueError("initial state shape does not match the dataset")
    return project_density_matrix(state)


def _validate_positive_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be at least one")
    return result


def _validate_positive_real(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_nonnegative_real(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _validate_open_unit_interval(value, name):
    result = _validate_positive_real(value, name)
    if result >= 1:
        raise ValueError(f"{name} must be strictly less than one")
    return result


def _as_float(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return float(value)


def _as_complex(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return complex(value)
