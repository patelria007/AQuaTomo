"""Complete local-Pauli measurement simulation with multinomial shots."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np

from .backend import adjoint, array_namespace, asarray, device_of, real_dtype, to_numpy
from .operators import measurement_unitary


@dataclass(frozen=True)
class MeasurementData:
    """Counts for local Pauli settings.

    Outcome indices are big-endian bit strings: bit 0 is the +1 eigenstate and
    bit 1 is the -1 eigenstate of the corresponding local Pauli operator.
    """

    n_qubits: int
    counts: Mapping[str, object]
    shots_per_setting: int

    @property
    def settings(self):
        return tuple(self.counts)

    @property
    def informationally_complete(self) -> bool:
        return set(self.settings) == set(complete_pauli_settings(self.n_qubits))

    def frequencies(self):
        return {key: value / self.shots_per_setting for key, value in self.counts.items()}


def complete_pauli_settings(n_qubits: int):
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    return tuple("".join(chars) for chars in product("XYZ", repeat=n_qubits))


def global_pauli_settings(n_qubits: int):
    """The three notebook-style settings; incomplete for more than one qubit."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    return tuple(axis * n_qubits for axis in "XYZ")


def pauli_probabilities(rho, setting: str):
    xp = array_namespace(rho)
    dim = rho.shape[-1]
    if rho.shape != (dim, dim) or len(setting) != dim.bit_length() - 1 or 2 ** len(setting) != dim:
        raise ValueError("rho and setting dimensions are inconsistent")
    unitary = measurement_unitary(setting, xp)
    rotated = adjoint(unitary, xp) @ rho @ unitary
    probabilities = xp.real(xp.diagonal(rotated))
    probabilities = xp.maximum(probabilities, xp.asarray(0.0, dtype=probabilities.dtype))
    total = xp.sum(probabilities)
    return probabilities / total


def simulate_pauli_measurements(
    rho,
    shots_per_setting: int,
    *,
    settings=None,
    rng=None,
) -> MeasurementData:
    """Draw multinomial counts and return them on the same backend/device.

    Random sampling is an explicit control-plane operation because multinomial
    RNG is outside the Python Array API Standard.  Born-rule linear algebra is
    performed natively; only the probability vector crosses to the seeded host
    generator, and counts are immediately transferred back.
    """

    if shots_per_setting < 1:
        raise ValueError("shots_per_setting must be positive")
    xp = array_namespace(rho)
    dim = rho.shape[-1]
    n_qubits = dim.bit_length() - 1
    if rho.shape != (dim, dim) or 2**n_qubits != dim:
        raise ValueError("rho must be a square 2^n by 2^n matrix")
    settings = complete_pauli_settings(n_qubits) if settings is None else tuple(settings)
    generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    counts = {}
    for setting in settings:
        probs = np.asarray(to_numpy(pauli_probabilities(rho, setting)), dtype=float)
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()
        sampled = generator.multinomial(shots_per_setting, probs)
        counts[setting] = asarray(sampled, xp, dtype=getattr(xp, "int64", None), device=device_of(rho))
    return MeasurementData(n_qubits=n_qubits, counts=counts, shots_per_setting=shots_per_setting)


def exact_pauli_measurements(rho, *, shots_per_setting: int = 1_000_000) -> MeasurementData:
    """Deterministic probability-weighted counts for numerical verification."""

    xp = array_namespace(rho)
    n_qubits = rho.shape[-1].bit_length() - 1
    counts = {}
    for setting in complete_pauli_settings(n_qubits):
        probabilities = pauli_probabilities(rho, setting)
        counts[setting] = probabilities * shots_per_setting
    return MeasurementData(n_qubits, counts, shots_per_setting)


def split_measurement_data(data: MeasurementData, validation_fraction=0.2, *, rng=None):
    """Split observed counts into independent train/validation partitions.

    A multivariate-hypergeometric draw partitions the already observed shots
    without resimulating the unknown state and preserves a fixed shot total for
    every setting.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    validation_shots = int(round(data.shots_per_setting * validation_fraction))
    validation_shots = min(max(validation_shots, 1), data.shots_per_setting - 1)
    train_shots = data.shots_per_setting - validation_shots
    generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    first = next(iter(data.counts.values()))
    xp = array_namespace(first)
    device = device_of(first)
    train_counts, validation_counts = {}, {}
    for setting, counts in data.counts.items():
        host_counts = np.asarray(to_numpy(counts), dtype=np.int64)
        held_out = generator.multivariate_hypergeometric(host_counts, validation_shots)
        remaining = host_counts - held_out
        train_counts[setting] = asarray(remaining, xp, dtype=getattr(xp, "int64", None), device=device)
        validation_counts[setting] = asarray(held_out, xp, dtype=getattr(xp, "int64", None), device=device)
    return (
        MeasurementData(data.n_qubits, train_counts, train_shots),
        MeasurementData(data.n_qubits, validation_counts, validation_shots),
    )
