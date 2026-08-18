"""Small compatibility layer for Python Array API-style namespaces.

The numerical kernels ask their input arrays for a namespace instead of
importing NumPy.  NumPy is used only as the default host/control-plane backend
and for deterministic random-number generation and serialization.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def array_namespace(*arrays: Any):
    """Return one common Array API namespace for ``arrays``.

    ``array_api_compat`` is used when installed.  The fallback supports modern
    Array API objects and regular NumPy arrays without making it mandatory.
    """

    arrays = tuple(a for a in arrays if a is not None)
    try:
        from array_api_compat import array_namespace as compat_namespace

        return compat_namespace(*arrays) if arrays else np
    except ImportError:
        namespaces = [a.__array_namespace__() for a in arrays if hasattr(a, "__array_namespace__")]
        if namespaces:
            first = namespaces[0]
            if any(ns is not first and getattr(ns, "__name__", None) != getattr(first, "__name__", None) for ns in namespaces[1:]):
                raise TypeError("All arrays must belong to the same Array API namespace")
            return first
        modules = {type(a).__module__.split(".")[0] for a in arrays}
        if modules <= {"numpy"} or not arrays:
            return np
        if len(modules) != 1:
            raise TypeError(f"Mixed array backends are unsupported: {sorted(modules)}")
        module = modules.pop()
        if module == "cupy":
            import cupy

            return cupy
        if module in {"jax", "jaxlib"}:
            import jax.numpy

            return jax.numpy
        raise TypeError(f"Cannot determine Array API namespace for module {module!r}")


def backend_name(xp_or_array: Any) -> str:
    xp = xp_or_array if hasattr(xp_or_array, "asarray") else array_namespace(xp_or_array)
    name = getattr(xp, "__name__", type(xp).__name__).lower()
    if "cupy" in name:
        return "cupy"
    if "jax" in name:
        return "jax"
    if "torch" in name:
        return "torch"
    if "numpy" in name:
        return "numpy"
    return name


def asarray(x: Any, xp, *, dtype=None, device=None):
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    try:
        return xp.asarray(x, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return xp.asarray(x, **kwargs)


def _creation(function, *args, dtype=None, device=None):
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    try:
        return function(*args, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return function(*args, **kwargs)


def zeros(shape, xp, *, dtype=None, device=None):
    return _creation(xp.zeros, shape, dtype=dtype, device=device)


def eye(dimension, xp, *, dtype=None, device=None):
    return _creation(xp.eye, dimension, dtype=dtype, device=device)


def arange(*args, xp, dtype=None, device=None):
    return _creation(xp.arange, *args, dtype=dtype, device=device)


def device_of(array: Any):
    return getattr(array, "device", None)


def to_numpy(array: Any) -> np.ndarray:
    """Explicit control-plane transfer used for RNG, logging, and files."""

    module = type(array).__module__.split(".")[0]
    if module == "cupy":
        return array.get()
    return np.asarray(array)


def real_dtype(xp):
    return getattr(xp, "float64", np.float64)


def complex_dtype(xp):
    return getattr(xp, "complex128", np.complex128)


def matrix_transpose(a, xp):
    if hasattr(xp.linalg, "matrix_transpose"):
        return xp.linalg.matrix_transpose(a)
    return xp.swapaxes(a, -1, -2)


def adjoint(a, xp):
    return xp.conj(matrix_transpose(a, xp))


def cumulative_sum(a, xp):
    if hasattr(xp, "cumulative_sum"):
        return xp.cumulative_sum(a)
    return xp.cumsum(a)


def kron(a, b, xp):
    """Array-API-only Kronecker product for two matrices."""

    a = xp.asarray(a)
    b = xp.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("kron currently accepts two matrices")
    product = a[:, None, :, None] * b[None, :, None, :]
    return xp.reshape(product, (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]))


def kron_all(matrices, xp):
    out = xp.ones((1, 1), dtype=matrices[0].dtype)
    for matrix in matrices:
        out = kron(out, matrix, xp)
    return out


def scalar(value: Any) -> float:
    return float(to_numpy(value))
