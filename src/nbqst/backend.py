"""Small compatibility layer for Python Array API-style namespaces.

The numerical kernels ask their input arrays for a namespace instead of
importing NumPy.  NumPy is used only as the default host/control-plane backend
and for deterministic random-number generation and serialization.
"""

from __future__ import annotations

import platform
import time
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


def _walk_values(value: Any):
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_values(item)
        return
    yield value


def synchronize(value: Any = None, *, xp=None):
    """Wait for queued backend work and return ``value`` unchanged.

    NumPy is synchronous.  CuPy requires a CUDA-stream barrier, while JAX
    arrays expose ``block_until_ready``.  Benchmark code should synchronize
    once immediately before starting a timer and once on the timed result.
    """

    values = tuple(_walk_values(value))
    if xp is None:
        arrays = tuple(item for item in values if hasattr(item, "shape"))
        xp = array_namespace(*arrays) if arrays else np
    name = backend_name(xp)
    if name == "cupy":
        xp.cuda.get_current_stream().synchronize()
    elif name == "jax":
        blocked = False
        for item in values:
            block = getattr(item, "block_until_ready", None)
            if block is not None:
                block()
                blocked = True
        if not blocked:
            try:
                import jax

                barrier = getattr(jax, "effects_barrier", None)
                if barrier is not None:
                    barrier()
            except ImportError:
                pass
    return value


def timed_call(function, *args, xp=None, synchronize_before=None, **kwargs):
    """Call ``function`` and return ``(result, elapsed_seconds)``.

    The timer uses nanosecond-resolution monotonic time and backend barriers,
    making the interval meaningful for eager CPU, CUDA/CuPy, and JAX work.
    Compilation is included unless the caller performs explicit warm-up calls.
    """

    synchronize(synchronize_before, xp=xp)
    started = time.perf_counter_ns()
    result = function(*args, **kwargs)
    synchronize(result, xp=xp)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
    return result, elapsed


def backend_runtime_metadata(xp) -> dict[str, Any]:
    """Return flat, CSV/JSON-friendly backend and primary-device metadata."""

    name = backend_name(xp)
    metadata: dict[str, Any] = {
        "backend": name,
        "device_platform": "cpu" if name == "numpy" else "unknown",
        "device_name": platform.processor() or platform.machine() or "CPU",
        "device_id": 0,
        "device_memory_bytes": "not_available",
        "compute_capability": "not_applicable",
        "backend_version": np.__version__ if name == "numpy" else "unknown",
        "synchronization": "host_synchronous" if name == "numpy" else "unknown",
    }
    if name == "cupy":
        device = xp.cuda.Device()
        properties = xp.cuda.runtime.getDeviceProperties(device.id)
        device_name = properties.get("name", "CUDA device")
        if isinstance(device_name, bytes):
            device_name = device_name.decode(errors="replace")
        metadata.update(
            {
                "device_platform": "cuda",
                "device_name": str(device_name),
                "device_id": int(device.id),
                "device_memory_bytes": int(properties.get("totalGlobalMem", 0)),
                "compute_capability": f"{properties.get('major', '?')}.{properties.get('minor', '?')}",
                "backend_version": getattr(xp, "__version__", "unknown"),
                "cuda_runtime_version": int(xp.cuda.runtime.runtimeGetVersion()),
                "cuda_driver_version": int(xp.cuda.runtime.driverGetVersion()),
                "synchronization": "cupy_current_stream_synchronize",
            }
        )
    elif name == "jax":
        import jax

        devices = jax.devices()
        device = devices[0] if devices else None
        memory = "not_available"
        if device is not None:
            try:
                stats = device.memory_stats() or {}
                memory = stats.get("bytes_limit", stats.get("bytes_reservable_limit", "not_available"))
            except (AttributeError, RuntimeError):
                pass
        metadata.update(
            {
                "device_platform": getattr(device, "platform", "unknown"),
                "device_name": getattr(device, "device_kind", str(device) if device is not None else "unknown"),
                "device_id": getattr(device, "id", 0),
                "device_memory_bytes": memory,
                "compute_capability": str(getattr(device, "compute_capability", "not_available")),
                "backend_version": getattr(jax, "__version__", "unknown"),
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "synchronization": "jax_block_until_ready",
            }
        )
        try:
            try:
                from jax.extend import backend as jax_backend

                client = jax_backend.get_backend()
            except (ImportError, AttributeError):
                client = jax.lib.xla_bridge.get_backend()
            metadata["accelerator_platform_version"] = getattr(client, "platform_version", "unknown")
        except (RuntimeError, AttributeError):
            metadata["accelerator_platform_version"] = "unavailable"
        try:
            import jaxlib

            metadata["jaxlib_version"] = getattr(jaxlib, "__version__", "unknown")
        except ImportError:
            metadata["jaxlib_version"] = "unknown"
    return metadata
