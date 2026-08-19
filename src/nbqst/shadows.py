"""Local-Pauli classical shadows for observable estimation.

The acquisition routine follows Huang, Kueng, and Preskill (2020): every
copy is measured in an independently and uniformly selected tensor-product
Pauli basis.  For a Pauli string of weight ``k``, the single-snapshot unbiased
estimator is ``3**k`` times the observed eigenvalue product when all supported
bases match, and zero otherwise.

Random basis selection and categorical sampling are explicit host control
operations. Born probabilities and observable post-processing stay in the
Array-API namespace of the state or shadow arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .backend import array_namespace, asarray, device_of, scalar, to_numpy
from .measurements import MeasurementData, pauli_probabilities
from .operators import pauli_string


PAULI_TO_CODE = {"X": 0, "Y": 1, "Z": 2}
CODE_TO_PAULI = ("X", "Y", "Z")


@dataclass(frozen=True)
class PauliObservable:
    """A real coefficient multiplying an ``I/X/Y/Z`` tensor product."""

    label: str
    coefficient: float = 1.0

    def __post_init__(self):
        normalized = self.label.upper()
        if not normalized or any(character not in "IXYZ" for character in normalized):
            raise ValueError("Pauli observable labels must be nonempty strings over I, X, Y, Z")
        object.__setattr__(self, "label", normalized)
        object.__setattr__(self, "coefficient", float(self.coefficient))

    @property
    def n_qubits(self) -> int:
        return len(self.label)

    @property
    def weight(self) -> int:
        return sum(character != "I" for character in self.label)

    def matrix(self, xp=np):
        return self.coefficient * pauli_string(self.label, xp)


@dataclass(frozen=True)
class ClassicalShadowData:
    """Random local bases and measured computational-basis outcome bits."""

    n_qubits: int
    basis_codes: object
    outcomes: object

    def __post_init__(self):
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.basis_codes.ndim != 2 or self.outcomes.ndim != 2:
            raise ValueError("basis_codes and outcomes must be rank-two arrays")
        if self.basis_codes.shape != self.outcomes.shape:
            raise ValueError("basis_codes and outcomes must have identical shapes")
        if self.basis_codes.shape[1] != self.n_qubits:
            raise ValueError("the array width must equal n_qubits")
        if self.basis_codes.shape[0] < 1:
            raise ValueError("at least one shadow snapshot is required")

    @property
    def num_snapshots(self) -> int:
        return int(self.basis_codes.shape[0])


@dataclass(frozen=True)
class ObservableEstimate:
    """Point estimate and an empirical uncertainty diagnostic."""

    observable: PauliObservable
    value: float
    standard_error: float
    samples: int
    method: str
    aggregation: str = "mean"

    @property
    def absolute_two_sigma(self) -> float:
        return 2.0 * self.standard_error


def observable_expectation(rho, observable: PauliObservable | str) -> float:
    """Return ``Re Tr(rho O)`` for one Pauli observable."""

    item = observable if isinstance(observable, PauliObservable) else PauliObservable(observable)
    if rho.shape != (2**item.n_qubits, 2**item.n_qubits):
        raise ValueError("state and observable dimensions are inconsistent")
    xp = array_namespace(rho)
    return scalar(xp.real(xp.trace(rho @ item.matrix(xp))))


def _outcome_bits(indices: np.ndarray, n_qubits: int) -> np.ndarray:
    shifts = np.arange(n_qubits - 1, -1, -1, dtype=np.int64)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)


class ClassicalShadowProtocol:
    """Acquire and query local-Pauli classical shadows.

    The class is stateless apart from its robust aggregation choice, making a
    protocol instance reusable across states and Array-API backends.
    """

    def __init__(self, *, median_of_means_groups: int = 1):
        if median_of_means_groups < 1:
            raise ValueError("median_of_means_groups must be positive")
        self.median_of_means_groups = int(median_of_means_groups)

    def acquire(self, rho, num_snapshots: int, *, rng=None) -> ClassicalShadowData:
        """Measure ``num_snapshots`` independently prepared copies of ``rho``."""

        if num_snapshots < 1:
            raise ValueError("num_snapshots must be positive")
        dimension = int(rho.shape[-1])
        n_qubits = dimension.bit_length() - 1
        if rho.shape != (dimension, dimension) or 2**n_qubits != dimension:
            raise ValueError("rho must be a square 2^n by 2^n matrix")

        generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        host_bases = generator.integers(0, 3, size=(num_snapshots, n_qubits), dtype=np.int8)
        settings = np.asarray(
            ["".join(CODE_TO_PAULI[int(code)] for code in row) for row in host_bases],
            dtype=f"U{n_qubits}",
        )
        host_outcomes = np.empty((num_snapshots, n_qubits), dtype=np.int8)
        for setting in np.unique(settings):
            positions = np.flatnonzero(settings == setting)
            probabilities = np.asarray(to_numpy(pauli_probabilities(rho, str(setting))), dtype=float)
            probabilities = np.maximum(probabilities, 0.0)
            probabilities /= probabilities.sum()
            sampled = generator.choice(dimension, size=len(positions), p=probabilities)
            host_outcomes[positions] = _outcome_bits(sampled, n_qubits)

        xp = array_namespace(rho)
        device = device_of(rho)
        return ClassicalShadowData(
            n_qubits=n_qubits,
            basis_codes=asarray(host_bases, xp, dtype=getattr(xp, "int8", None), device=device),
            outcomes=asarray(host_outcomes, xp, dtype=getattr(xp, "int8", None), device=device),
        )

    def _snapshot_values(self, shadow: ClassicalShadowData, observable: PauliObservable):
        if observable.n_qubits != shadow.n_qubits:
            raise ValueError("shadow data and observable use different qubit counts")
        xp = array_namespace(shadow.basis_codes, shadow.outcomes)
        support = [index for index, character in enumerate(observable.label) if character != "I"]
        if not support:
            return xp.ones((shadow.num_snapshots,), dtype=getattr(xp, "float64", None)) * observable.coefficient

        targets = asarray(
            [PAULI_TO_CODE[observable.label[index]] for index in support],
            xp,
            dtype=getattr(xp, "int8", None),
            device=device_of(shadow.basis_codes),
        )
        selected_bases = xp.stack(tuple(shadow.basis_codes[:, index] for index in support), axis=1)
        selected_outcomes = xp.stack(tuple(shadow.outcomes[:, index] for index in support), axis=1)
        matches = xp.all(selected_bases == targets[None, :], axis=1)
        eigenvalue_product = xp.prod(1 - 2 * selected_outcomes, axis=1)
        scale = observable.coefficient * (3**observable.weight)
        return scale * eigenvalue_product * matches

    def estimate(
        self,
        shadow: ClassicalShadowData,
        observable: PauliObservable | str,
    ) -> ObservableEstimate:
        """Estimate one Pauli expectation from a reusable shadow data set."""

        item = observable if isinstance(observable, PauliObservable) else PauliObservable(observable)
        values = self._snapshot_values(shadow, item)
        xp = array_namespace(values)
        sample_count = shadow.num_snapshots
        groups = min(self.median_of_means_groups, sample_count)
        if groups == 1:
            point = xp.mean(values)
            standard_error = (
                xp.std(values, ddof=1) / xp.sqrt(xp.asarray(sample_count, dtype=values.dtype))
                if sample_count > 1
                else xp.asarray(0.0, dtype=values.dtype)
            )
            aggregation = "mean"
        else:
            usable = sample_count - sample_count % groups
            block_means = xp.mean(xp.reshape(values[:usable], (groups, usable // groups)), axis=1)
            point = xp.median(block_means)
            standard_error = (
                xp.std(block_means, ddof=1) / xp.sqrt(xp.asarray(groups, dtype=block_means.dtype))
                if groups > 1
                else xp.asarray(0.0, dtype=block_means.dtype)
            )
            aggregation = f"median_of_means_{groups}_groups"
        return ObservableEstimate(
            observable=item,
            value=scalar(point),
            standard_error=scalar(standard_error),
            samples=sample_count,
            method="classical_shadow",
            aggregation=aggregation,
        )

    def estimate_many(
        self,
        shadow: ClassicalShadowData,
        observables: Iterable[PauliObservable | str],
    ) -> tuple[ObservableEstimate, ...]:
        """Query several observables without acquiring new measurements."""

        return tuple(self.estimate(shadow, observable) for observable in observables)


def estimate_observable_from_measurements(
    data: MeasurementData,
    observable: PauliObservable | str,
) -> ObservableEstimate:
    """Directly estimate a Pauli string from one compatible QST setting.

    Identity positions are measured in Z and ignored. This estimator is useful
    as a direct-measurement reference; full QST still paid for every ``3**n``
    setting in ``data``.
    """

    item = observable if isinstance(observable, PauliObservable) else PauliObservable(observable)
    if item.n_qubits != data.n_qubits:
        raise ValueError("measurement data and observable use different qubit counts")
    setting = "".join(character if character != "I" else "Z" for character in item.label)
    try:
        counts = data.counts[setting]
    except KeyError as error:
        raise ValueError(f"measurement data do not contain compatible setting {setting}") from error
    xp = array_namespace(counts)
    dimension = 2**data.n_qubits
    indices = xp.arange(dimension)
    signs = xp.ones((dimension,), dtype=getattr(xp, "float64", None))
    for qubit, character in enumerate(item.label):
        if character != "I":
            bit = (indices // (2 ** (data.n_qubits - qubit - 1))) % 2
            signs = signs * (1 - 2 * bit)
    total = xp.sum(counts)
    point = item.coefficient * xp.sum(counts * signs) / total
    second_moment = item.coefficient**2
    variance = xp.maximum(second_moment - point * point, xp.asarray(0.0))
    standard_error = xp.sqrt(variance / total)
    return ObservableEstimate(
        observable=item,
        value=scalar(point),
        standard_error=scalar(standard_error),
        samples=int(data.shots_per_setting),
        method="direct_pauli_setting",
    )
