from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from logic.Optimizer import Optimizer, SGD
from logic.activationFunction import UnaryActivation, relu, softmax
from logic.lossFunction import LossFunction, cross_entropy
from structure.InputLayer import InputLayer
from structure.Node import Node
from structure.hiddenLayer import HiddenLayer
from structure.outputLayer import OutputLayer


FloatArray = NDArray[np.float64]


@dataclass
class NeuralNetwork:
    """
    Minimal MLP built from the modular layer components.

    The network exposes high-level training utilities (`train_step`, `fit`) while delegating the
    math-heavy pieces to the individual modules for clarity.
    """

    input_layer: InputLayer
    hidden_layers: List[HiddenLayer]
    output_layer: OutputLayer
    loss: LossFunction = field(default=cross_entropy)
    optimizer: Optimizer = field(default_factory=lambda: SGD(learning_rate=0.1))

    def forward(self, input_batch: FloatArray) -> FloatArray:
        layer_outputs = self.input_layer.forward(input_batch)
        for hidden_layer in self.hidden_layers:
            layer_outputs = hidden_layer.forward(layer_outputs)
        return self.output_layer.forward(layer_outputs)

    def backward(self, predictions: FloatArray, targets: FloatArray) -> None:
        loss_gradients = self.loss.derivative(predictions, targets)
        should_apply_activation_derivative = not (
            self.loss is cross_entropy and self.output_layer.activation is softmax
        )
        incoming_gradient = self.output_layer.backward(
            loss_gradients,
            apply_activation_derivative=should_apply_activation_derivative,
        )
        for hidden_layer in reversed(self.hidden_layers):
            incoming_gradient = hidden_layer.backward(incoming_gradient)

    def parameter_grad_pairs(self) -> List[tuple[FloatArray, FloatArray]]:
        parameter_gradient_pairs: List[tuple[FloatArray, FloatArray]] = []
        for hidden_layer in self.hidden_layers:
            parameter_gradient_pairs.extend(hidden_layer.parameter_pairs())
        parameter_gradient_pairs.extend(self.output_layer.parameter_pairs())
        return parameter_gradient_pairs

    def zero_gradients(self) -> None:
        for hidden_layer in self.hidden_layers:
            hidden_layer.parameters.zero_gradients()
        self.output_layer.parameters.zero_gradients()

    def train_step(self, training_inputs: FloatArray, training_targets: FloatArray) -> float:
        predictions = self.forward(training_inputs)
        loss_value = self.loss(predictions, training_targets)
        self.backward(predictions, training_targets)
        self.optimizer.step(self.parameter_grad_pairs())
        self.zero_gradients()
        return loss_value

    def fit(
        self,
        inputs: FloatArray,
        targets: FloatArray,
        epochs: int = 100,
        batch_size: int = 32,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> List[float]:
        generator = rng or np.random.default_rng()
        num_samples = inputs.shape[0]
        history: List[float] = []

        for _ in range(epochs):
            sample_indices = np.arange(num_samples)
            if shuffle:
                generator.shuffle(sample_indices)
            batch_index_list = [
                sample_indices[start : start + batch_size]
                for start in range(0, num_samples, batch_size)
            ]
            epoch_loss_values: List[float] = []
            for batch_indices in batch_index_list:
                batch_inputs = inputs[batch_indices]
                batch_targets = targets[batch_indices]
                batch_loss = self.train_step(batch_inputs, batch_targets)
                epoch_loss_values.append(batch_loss)
            history.append(float(np.mean(epoch_loss_values)))
        return history

    def predict(self, input_batch: FloatArray) -> FloatArray:
        probability_distribution = self.forward(input_batch)
        return self.output_layer.predict(probability_distribution)


def gradient_check(
    network: NeuralNetwork,
    inputs: FloatArray,
    targets: FloatArray,
    epsilon: float = 1e-5,
) -> float:
    """
    Finite-difference gradient check to spot implementation bugs.

    Returns the maximum absolute difference between analytic and numerical gradients.
    Smaller values (<1e-6) typically indicate correct derivatives.
    """

    predictions = network.forward(inputs)
    network.backward(predictions, targets)

    parameter_pairs = network.parameter_grad_pairs()
    analytic_gradients = [grad.copy() for _, grad in parameter_pairs]
    parameter_arrays = [param for param, _ in parameter_pairs]

    max_difference = 0.0
    for parameters, analytic in zip(parameter_arrays, analytic_gradients):
        iterator = np.nditer(parameters, flags=["multi_index"], op_flags=["readwrite"])
        while not iterator.finished:
            idx = iterator.multi_index
            original_value = parameters[idx]

            parameters[idx] = original_value + epsilon
            plus_loss = network.loss(network.forward(inputs), targets)

            parameters[idx] = original_value - epsilon
            minus_loss = network.loss(network.forward(inputs), targets)

            parameters[idx] = original_value
            numerical_derivative = (plus_loss - minus_loss) / (2.0 * epsilon)

            difference = float(abs(numerical_derivative - analytic[idx]))
            if difference > max_difference:
                max_difference = difference
            iterator.iternext()

    network.zero_gradients()
    return max_difference


def make_toy_classification_dataset(
    samples_per_class: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray]:
    """
    Two moons-style synthetic dataset for quick experimentation.

    Generates two interleaving semicircles and returns one-hot encoded targets.
    """

    generator = rng or np.random.default_rng(0)
    angles = generator.uniform(0.0, np.pi, size=samples_per_class)
    radius = 1.0 + 0.1 * generator.normal(size=samples_per_class)
    class_a_samples = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    angles_b = angles + np.pi
    radius_b = 1.0 + 0.1 * generator.normal(size=samples_per_class)
    class_b_samples = np.stack(
        [
            radius_b * np.cos(angles_b) + 0.5,
            radius_b * np.sin(angles_b) - 0.2,
        ],
        axis=1,
    )

    inputs = np.vstack([class_a_samples, class_b_samples]).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(samples_per_class, dtype=np.int64), np.ones(samples_per_class, dtype=np.int64)]
    )
    targets = np.eye(2, dtype=np.float64)[labels]
    return inputs, targets


def build_demo_network(
    input_dim: int,
    hidden_units: Sequence[int],
    output_dim: int,
    activation: UnaryActivation = relu,
    rng: np.random.Generator | None = None,
) -> NeuralNetwork:
    """
    Convenience factory wiring the modular pieces together.

    `hidden_units` allows stacking multiple hidden layers without changing the surrounding code.
    """

    generator = rng or np.random.default_rng(1)
    input_layer = InputLayer(input_dim=input_dim)

    hidden_layers: List[HiddenLayer] = []
    previous_unit_count = input_dim
    for units in hidden_units:
        hidden_parameters = Node.initialize(previous_unit_count, units, rng=generator)
        hidden_layers.append(HiddenLayer(parameters=hidden_parameters, activation=activation))
        previous_unit_count = units

    output_parameters = Node.initialize(previous_unit_count, output_dim, rng=generator)
    output_layer = OutputLayer(parameters=output_parameters, activation=softmax)

    return NeuralNetwork(
        input_layer=input_layer,
        hidden_layers=hidden_layers,
        output_layer=output_layer,
        loss=cross_entropy,
        optimizer=SGD(learning_rate=0.1, max_grad_norm=1.0),
    )


def demo_training_epoch() -> None:
    """
    Example usage training on a synthetic dataset.

    Prints epoch-level losses and performs a gradient check on the initial parameters for sanity.
    """

    inputs, targets = make_toy_classification_dataset(samples_per_class=200)
    network = build_demo_network(input_dim=inputs.shape[1], hidden_units=(16, 16), output_dim=2)

    gradient_difference = gradient_check(network, inputs[:5], targets[:5])
    print(f"Gradient check (max diff): {gradient_difference:.6e}")

    loss_history = network.fit(inputs, targets, epochs=50, batch_size=32)
    print(f"Final training loss: {loss_history[-1]:.4f}")

    predictions = network.predict(inputs)
    accuracy = np.mean(predictions.flatten() == np.argmax(targets, axis=1))
    print(f"Training accuracy: {accuracy:.3f}")


__all__ = [
    "NeuralNetwork",
    "gradient_check",
    "make_toy_classification_dataset",
    "build_demo_network",
    "demo_training_epoch",
]


if __name__ == "__main__":
    demo_training_epoch()

