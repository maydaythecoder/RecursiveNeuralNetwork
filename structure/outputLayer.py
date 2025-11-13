from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from logic.activationFunction import UnaryActivation, softmax
from structure.Node import Node


FloatArray = NDArray[np.float64]


@dataclass
class OutputLayer:
    """
    Dense readout layer with configurable activation.

    Default activation is softmax for multi-class classification. Swap in identity or sigmoid when
    doing regression or binary outputs.
    """

    parameters: Node
    activation: UnaryActivation = softmax
    cached_input_batch: FloatArray | None = field(default=None, init=False)
    cached_logit_batch: FloatArray | None = field(default=None, init=False)
    cached_activation: FloatArray | None = field(default=None, init=False)

    def forward(self, input_batch: FloatArray) -> FloatArray:
        self.cached_input_batch = input_batch
        logits = input_batch @ self.parameters.weight_matrix + self.parameters.bias_vector
        self.cached_logit_batch = logits
        activation_output = self.activation(logits)
        self.cached_activation = activation_output
        return activation_output

    def backward(
        self,
        output_gradient: FloatArray,
        *,
        apply_activation_derivative: bool = True,
    ) -> FloatArray:
        if self.cached_input_batch is None or self.cached_activation is None:
            raise RuntimeError("forward must be called before backward.")

        sample_count = output_gradient.shape[0]
        if apply_activation_derivative:
            activation_derivative = self.activation.derivative(self.cached_activation)
            post_activation_gradient = output_gradient * activation_derivative
        else:
            post_activation_gradient = output_gradient

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

    def predict(self, outputs: FloatArray | None = None) -> FloatArray:
        scores = outputs if outputs is not None else self.cached_activation
        if scores is None:
            raise RuntimeError("No cached outputs to derive predictions from.")
        return np.argmax(scores, axis=1, keepdims=True)


__all__ = ["OutputLayer"]

