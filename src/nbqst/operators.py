"""Qubit operators and local Pauli measurement bases."""

from __future__ import annotations

from itertools import product

from .backend import asarray, complex_dtype, kron_all


def pauli(label: str, xp):
    data = {
        "I": [[1, 0], [0, 1]],
        "X": [[0, 1], [1, 0]],
        "Y": [[0, -1j], [1j, 0]],
        "Z": [[1, 0], [0, -1]],
    }
    try:
        return asarray(data[label], xp, dtype=complex_dtype(xp))
    except KeyError as exc:
        raise ValueError(f"Unknown Pauli label: {label!r}") from exc


def pauli_string(label: str, xp):
    if not label:
        raise ValueError("Pauli string must be nonempty")
    return kron_all([pauli(ch, xp) for ch in label], xp)


def pauli_labels(n_qubits: int):
    return tuple("".join(chars) for chars in product("IXYZ", repeat=n_qubits))


def single_qubit_measurement_unitary(axis: str, xp):
    """Columns are the +1 and -1 eigenvectors for ``axis``."""

    inv_sqrt_2 = 2.0**-0.5
    data = {
        "Z": [[1, 0], [0, 1]],
        "X": [[inv_sqrt_2, inv_sqrt_2], [inv_sqrt_2, -inv_sqrt_2]],
        "Y": [[inv_sqrt_2, inv_sqrt_2], [1j * inv_sqrt_2, -1j * inv_sqrt_2]],
    }
    try:
        return asarray(data[axis], xp, dtype=complex_dtype(xp))
    except KeyError as exc:
        raise ValueError(f"Measurement axis must be X, Y, or Z, got {axis!r}") from exc


def measurement_unitary(setting: str, xp):
    if not setting or any(ch not in "XYZ" for ch in setting):
        raise ValueError("A measurement setting must be a nonempty string over X, Y, Z")
    return kron_all([single_qubit_measurement_unitary(ch, xp) for ch in setting], xp)

