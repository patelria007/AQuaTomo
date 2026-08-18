"""Dependency-light neural-network quantum-state reconstruction.

The estimator follows the feed-forward, Cholesky-output strategy studied by
Koutny et al. (arXiv:2206.06736): measured frequencies are mapped to a real
parameter vector describing a complex lower-triangular factor ``L`` and the
reported state is ``rho = L L^dagger / trace(L L^dagger)``.  Consequently every
inference result is Hermitian, positive semidefinite, and trace one.

Training uses a compact NumPy implementation of Adam so NumPy remains the only
mandatory dependency.  Inference performs only Array-API-style operations and
therefore follows the backend/device of the measurement-count arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .backend import adjoint, array_namespace, asarray, complex_dtype, device_of, eye, to_numpy
from .measurements import MeasurementData, complete_pauli_settings


@dataclass(frozen=True)
class NeuralTomographyModel:
    """Weights and metadata for a fully connected tomography estimator."""

    n_qubits: int
    settings: tuple[str, ...]
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    @property
    def dimension(self) -> int:
        return 2**self.n_qubits

    @property
    def input_size(self) -> int:
        return len(self.settings) * self.dimension

    @property
    def output_size(self) -> int:
        return self.dimension**2

    def validate(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError("weights and biases must contain the same nonzero number of layers")
        expected_in = self.input_size
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if weight.ndim != 2 or bias.ndim != 1:
                raise ValueError(f"layer {index} must contain a matrix and a vector")
            if weight.shape[0] != expected_in or weight.shape[1] != bias.shape[0]:
                raise ValueError(f"inconsistent shapes in layer {index}")
            expected_in = weight.shape[1]
        if expected_in != self.output_size:
            raise ValueError(f"output layer must have {self.output_size} neurons")


def measurement_features(data: MeasurementData, *, settings: Sequence[str] | None = None):
    """Flatten normalized outcome frequencies in a deterministic setting order."""

    settings = tuple(complete_pauli_settings(data.n_qubits) if settings is None else settings)
    missing = set(settings) - set(data.settings)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Neural reconstruction is missing {len(missing)} settings ({preview})")
    first = next(iter(data.counts.values()))
    xp = array_namespace(first)
    vectors = []
    for setting in settings:
        counts = data.counts[setting]
        vectors.append(counts / xp.sum(counts))
    concatenate = getattr(xp, "concat", None) or xp.concatenate
    return concatenate(tuple(vectors), axis=0)


def _factor_basis(dimension: int) -> np.ndarray:
    """Return basis matrices for d^2 real lower-triangular parameters."""

    basis = []
    for row in range(dimension):
        matrix = np.zeros((dimension, dimension), dtype=np.complex128)
        matrix[row, row] = 1.0
        basis.append(matrix)
    for row in range(1, dimension):
        for column in range(row):
            real = np.zeros((dimension, dimension), dtype=np.complex128)
            imaginary = np.zeros((dimension, dimension), dtype=np.complex128)
            real[row, column] = 1.0
            imaginary[row, column] = 1.0j
            basis.extend((real, imaginary))
    return np.stack(basis)


def cholesky_parameters_to_density(parameters, *, dimension: int):
    """Convert d^2 real Cholesky parameters into a physical density matrix."""

    xp = array_namespace(parameters)
    if parameters.ndim != 1 or parameters.shape[0] != dimension**2:
        raise ValueError(f"parameters must have shape ({dimension**2},)")
    basis = asarray(
        _factor_basis(dimension),
        xp,
        dtype=complex_dtype(xp),
        device=device_of(parameters),
    )
    factor = xp.sum(parameters[:, None, None] * basis, axis=0)
    gram = factor @ adjoint(factor, xp)
    trace = xp.real(xp.trace(gram))
    epsilon = xp.asarray(1e-15, dtype=trace.dtype)
    identity = eye(dimension, xp, dtype=complex_dtype(xp), device=device_of(parameters))
    safe_gram = gram + xp.asarray(trace <= epsilon, dtype=trace.dtype) * identity
    return safe_gram / xp.real(xp.trace(safe_gram))


def density_to_cholesky_parameters(rho, *, regularization: float = 1e-8) -> np.ndarray:
    """Encode a target state as d^2 real lower-triangular parameters."""

    matrix = np.asarray(to_numpy(rho), dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rho must be a square matrix")
    if not 0.0 < regularization < 1.0:
        raise ValueError("regularization must lie strictly between zero and one")
    dimension = matrix.shape[0]
    matrix = (matrix + matrix.conj().T) / 2.0
    matrix = (1.0 - regularization) * matrix + regularization * np.eye(dimension) / dimension
    matrix /= np.trace(matrix).real
    factor = np.linalg.cholesky(matrix)
    values = [float(factor[index, index].real) for index in range(dimension)]
    for row in range(1, dimension):
        for column in range(row):
            values.extend((float(factor[row, column].real), float(factor[row, column].imag)))
    return np.asarray(values, dtype=np.float64)


def _forward(features, weights, biases, xp):
    activation = features
    for index, (host_weight, host_bias) in enumerate(zip(weights, biases)):
        weight = asarray(host_weight, xp, dtype=features.dtype, device=device_of(features))
        bias = asarray(host_bias, xp, dtype=features.dtype, device=device_of(features))
        activation = activation @ weight + bias
        activation = xp.tanh(activation) if index == len(weights) - 1 else xp.maximum(activation, 0.0)
    return activation


def neural_state_reconstruction(data: MeasurementData, model: NeuralTomographyModel):
    """Reconstruct one physical state from measurement data using a trained MLP.

    This is the standalone neural estimator requested for the three-method
    comparison.  The model is dimension- and measurement-design-specific; it
    rejects incompatible data rather than silently reordering features.
    """

    model.validate()
    if data.n_qubits != model.n_qubits:
        raise ValueError("measurement data and neural model use different qubit counts")
    features = measurement_features(data, settings=model.settings)
    xp = array_namespace(features)
    parameters = _forward(features, model.weights, model.biases, xp)
    return cholesky_parameters_to_density(parameters, dimension=model.dimension)


def _training_arrays(states: Sequence[object], datasets: Sequence[MeasurementData], settings):
    if len(states) != len(datasets) or len(states) < 2:
        raise ValueError("states and datasets must have the same length of at least two")
    features, targets = [], []
    for state, data in zip(states, datasets):
        features.append(np.asarray(to_numpy(measurement_features(data, settings=settings)), dtype=np.float64))
        targets.append(density_to_cholesky_parameters(state))
    return np.stack(features), np.stack(targets)


def train_neural_reconstructor(
    states: Sequence[object],
    datasets: Sequence[MeasurementData],
    *,
    hidden_layers: Sequence[int] = (128, 128),
    epochs: int = 400,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    patience: int = 50,
    seed: int = 7,
    return_history: bool = False,
):
    """Train a ReLU/tanh multilayer perceptron with Adam using only NumPy.

    The loss is mean-squared error on Cholesky parameters, matching the core
    supervised objective in the attached paper.  A deterministic validation
    split and early stopping restore the best observed weights.
    """

    if not states:
        raise ValueError("at least one training state is required")
    n_qubits = datasets[0].n_qubits
    if any(data.n_qubits != n_qubits for data in datasets):
        raise ValueError("all training datasets must use the same qubit count")
    if any(width < 1 for width in hidden_layers):
        raise ValueError("hidden-layer widths must be positive")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("invalid optimizer configuration")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")

    settings = complete_pauli_settings(n_qubits)
    x, y = _training_arrays(states, datasets, settings)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    validation_size = min(max(int(round(len(x) * validation_fraction)), 1), len(x) - 1)
    validation_indices, training_indices = order[:validation_size], order[validation_size:]
    x_train, y_train = x[training_indices], y[training_indices]
    x_validation, y_validation = x[validation_indices], y[validation_indices]

    layer_sizes = (x.shape[1], *(int(width) for width in hidden_layers), y.shape[1])
    weights = []
    biases = []
    for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:]):
        scale = np.sqrt(2.0 / fan_in)
        weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
        biases.append(np.zeros(fan_out, dtype=np.float64))

    first_moment_w = [np.zeros_like(value) for value in weights]
    second_moment_w = [np.zeros_like(value) for value in weights]
    first_moment_b = [np.zeros_like(value) for value in biases]
    second_moment_b = [np.zeros_like(value) for value in biases]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    best_loss = np.inf
    best_parameters = None
    stale_epochs = 0
    history = []
    step = 0

    for epoch in range(epochs):
        shuffled = rng.permutation(len(x_train))
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            inputs, targets = x_train[indices], y_train[indices]
            activations = [inputs]
            preactivations = []
            value = inputs
            for layer, (weight, bias) in enumerate(zip(weights, biases)):
                preactivation = value @ weight + bias
                preactivations.append(preactivation)
                value = np.tanh(preactivation) if layer == len(weights) - 1 else np.maximum(preactivation, 0.0)
                activations.append(value)

            gradient = 2.0 * (activations[-1] - targets) / (len(inputs) * targets.shape[1])
            gradient *= 1.0 - activations[-1] ** 2
            gradients_w = [None] * len(weights)
            gradients_b = [None] * len(biases)
            for layer in range(len(weights) - 1, -1, -1):
                gradients_w[layer] = activations[layer].T @ gradient
                gradients_b[layer] = np.sum(gradient, axis=0)
                if layer:
                    gradient = (gradient @ weights[layer].T) * (preactivations[layer - 1] > 0.0)

            step += 1
            for layer in range(len(weights)):
                first_moment_w[layer] = beta1 * first_moment_w[layer] + (1.0 - beta1) * gradients_w[layer]
                second_moment_w[layer] = beta2 * second_moment_w[layer] + (1.0 - beta2) * gradients_w[layer] ** 2
                first_moment_b[layer] = beta1 * first_moment_b[layer] + (1.0 - beta1) * gradients_b[layer]
                second_moment_b[layer] = beta2 * second_moment_b[layer] + (1.0 - beta2) * gradients_b[layer] ** 2
                corrected_w = first_moment_w[layer] / (1.0 - beta1**step)
                variance_w = second_moment_w[layer] / (1.0 - beta2**step)
                corrected_b = first_moment_b[layer] / (1.0 - beta1**step)
                variance_b = second_moment_b[layer] / (1.0 - beta2**step)
                weights[layer] -= learning_rate * corrected_w / (np.sqrt(variance_w) + epsilon)
                biases[layer] -= learning_rate * corrected_b / (np.sqrt(variance_b) + epsilon)

        validation_prediction = _forward(x_validation, weights, biases, np)
        validation_loss = float(np.mean((validation_prediction - y_validation) ** 2))
        history.append({"epoch": epoch + 1, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_parameters = ([value.copy() for value in weights], [value.copy() for value in biases])
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    weights, biases = best_parameters
    model = NeuralTomographyModel(
        n_qubits=n_qubits,
        settings=tuple(settings),
        weights=tuple(weights),
        biases=tuple(biases),
    )
    return (model, history) if return_history else model


def save_neural_model(path: str | Path, model: NeuralTomographyModel) -> None:
    """Persist a trained model in a portable NumPy archive."""

    model.validate()
    payload = {
        "n_qubits": np.asarray(model.n_qubits),
        "settings": np.asarray(model.settings),
        "layers": np.asarray(len(model.weights)),
    }
    for index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        payload[f"weight_{index}"] = weight
        payload[f"bias_{index}"] = bias
    np.savez_compressed(Path(path), **payload)


def load_neural_model(path: str | Path) -> NeuralTomographyModel:
    """Load a model written by :func:`save_neural_model`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        layers = int(archive["layers"])
        model = NeuralTomographyModel(
            n_qubits=int(archive["n_qubits"]),
            settings=tuple(str(value) for value in archive["settings"]),
            weights=tuple(np.asarray(archive[f"weight_{index}"]) for index in range(layers)),
            biases=tuple(np.asarray(archive[f"bias_{index}"]) for index in range(layers)),
        )
    model.validate()
    return model
