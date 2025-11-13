from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ParameterGradientPair = Tuple[FloatArray, FloatArray]


class Optimizer(Protocol):
    """Optimizer interface that mutates parameters in place."""

    def step(self, parameter_gradient_pairs: Iterable[ParameterGradientPair]) -> None:
        raise NotImplementedError


@dataclass
class SGD:
    """
    Classic stochastic gradient descent with optional gradient clipping.

    The implementation assumes upstream code already averages gradients over the batch.
    """

    learning_rate: float = 0.01
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when supplied")

    def step(self, parameter_gradient_pairs: Iterable[ParameterGradientPair]) -> None:
        for parameter_values, gradients in parameter_gradient_pairs:
            safe_gradients = gradients
            if self.max_grad_norm is not None:
                safe_gradients = self._clip_gradients(gradients)
            parameter_values -= self.learning_rate * safe_gradients

    def _clip_gradients(self, gradients: FloatArray) -> FloatArray:
        norm = np.linalg.norm(gradients)
        if norm == 0.0 or norm <= self.max_grad_norm:  # type: ignore[operator]
            return gradients
        scale = self.max_grad_norm / norm  # type: ignore[operator]
        return gradients * scale


__all__ = [
    "Optimizer",
    "SGD",
]

