from __future__ import annotations
from dataclasses import dataclass, field
from numpy.typing import NDArray

from logic.activationFunction import UnaryActivation
from structure.Node import Node
import numpy as np


FloatArray = NDArray[np.float64]


@dataclass
class HiddenLayer:
    """
    Dense layer with activation.

    `Node` carries the trainable parameters, enabling the optimizer module to receive a unified view
    of `(parameters, gradients)` pairs across layers.
    """

    activation: UnaryActivation
    parameters: Node
    cached_input_batch: FloatArray | None = field(default=None, init=False)
    cached_weighted_sum: FloatArray | None = field(default=None, init=False)
    cached_activation: FloatArray | None = field(default=None, init=False)

    def forward(self, input_batch: FloatArray) -> FloatArray:
        self.cached_input_batch = input_batch
        weighted_sum = input_batch @ self.parameters.weight_matrix + self.parameters.bias_vector
        self.cached_weighted_sum = weighted_sum
        activation_output = self.activation(weighted_sum)
        self.cached_activation = activation_output
        return activation_output

    def backward(self, output_gradient: FloatArray) -> FloatArray:
        if self.cached_input_batch is None or self.cached_activation is None:
            raise RuntimeError("forward must be called before backward.")

        sample_count = output_gradient.shape[0]
        activation_derivative = self.activation.derivative(self.cached_activation)
        post_activation_gradient = output_gradient * activation_derivative

        self.parameters.weight_gradients = (
            self.cached_input_batch.T @ post_activation_gradient
        ) / sample_count
        self.parameters.bias_gradients = (
            np.sum(post_activation_gradient, axis=0, keepdims=True) / sample_count
        )

        return post_activation_gradient @ self.parameters.weight_matrix.T

    def parameter_pairs(self) -> list[tuple[FloatArray, FloatArray]]:
        return [
            (self.parameters.weight_matrix, self.parameters.weight_gradients),
            (self.parameters.bias_vector, self.parameters.bias_gradients),
        ]


__all__ = ["HiddenLayer"]

