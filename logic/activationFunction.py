from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class Activation(Protocol):
    """Interface for activation primitives providing forward and derivative calls."""

    def __call__(self, input_batch: FloatArray) -> FloatArray:
        raise NotImplementedError

    def derivative(self, activated_output: FloatArray) -> FloatArray:
        raise NotImplementedError


@dataclass(frozen=True)
class UnaryActivation:
    """
    Helper to keep forward and derivative logic together.

    Storing `forward` and `derivative` enables re-use across layers without coupling to layer
    internals.
    """

    forward_fn: Callable[[FloatArray], FloatArray]
    derivative_fn: Callable[[FloatArray], FloatArray]

    def __call__(self, input_batch: FloatArray) -> FloatArray:
        return self.forward_fn(input_batch)

    def derivative(self, activated_output: FloatArray) -> FloatArray:
        return self.derivative_fn(activated_output)


def _sigmoid_forward(input_batch: FloatArray) -> FloatArray:
    # SAFETY: Clip avoids overflow when exponentiating large magnitude inputs.
    clipped = np.clip(input_batch, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sigmoid_backward(activated_output: FloatArray) -> FloatArray:
    return activated_output * (1.0 - activated_output)


sigmoid: UnaryActivation = UnaryActivation(
    forward_fn=_sigmoid_forward,
    derivative_fn=_sigmoid_backward,
)


def _relu_forward(input_batch: FloatArray) -> FloatArray:
    return np.maximum(0.0, input_batch)


def _relu_backward(activated_output: FloatArray) -> FloatArray:
    return (activated_output > 0.0).astype(np.float64)


relu: UnaryActivation = UnaryActivation(
    forward_fn=_relu_forward,
    derivative_fn=_relu_backward,
)


def _softmax_forward(input_batch: FloatArray) -> FloatArray:
    shifted = input_batch - np.max(input_batch, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _softmax_backward(activated_output: FloatArray) -> FloatArray:
    """
    Gradient of the softmax output with respect to its inputs.

    Returns Jacobian diagonals flattened to match upstream gradient expectations.
    Each row corresponds to the diagonal entries for a sample's softmax vector, which is
    sufficient when paired with cross-entropy loss (full Jacobian rarely needed).
    """
    return activated_output * (1.0 - activated_output)


softmax: UnaryActivation = UnaryActivation(
    forward_fn=_softmax_forward,
    derivative_fn=_softmax_backward,
)


__all__ = [
    "Activation",
    "UnaryActivation",
    "sigmoid",
    "relu",
    "softmax",
]

