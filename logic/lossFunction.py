from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class Loss(Protocol):
    """Interface for loss primitives mirroring activation structure."""

    def __call__(self, prediction_batch: FloatArray, target_batch: FloatArray) -> float:
        raise NotImplementedError

    def derivative(self, prediction_batch: FloatArray, target_batch: FloatArray) -> FloatArray:
        raise NotImplementedError


@dataclass(frozen=True)
class LossFunction:
    forward_fn: Callable[[FloatArray, FloatArray], float]
    derivative_fn: Callable[[FloatArray, FloatArray], FloatArray]

    def __call__(self, prediction_batch: FloatArray, target_batch: FloatArray) -> float:
        return float(self.forward_fn(prediction_batch, target_batch))

    def derivative(self, prediction_batch: FloatArray, target_batch: FloatArray) -> FloatArray:
        return self.derivative_fn(prediction_batch, target_batch)


def _mean_squared_error(prediction_batch: FloatArray, target_batch: FloatArray) -> float:
    prediction_error = prediction_batch - target_batch
    return float(np.mean(np.square(prediction_error)))


def _mean_squared_error_grad(prediction_batch: FloatArray, target_batch: FloatArray) -> FloatArray:
    sample_count = prediction_batch.shape[0]
    return (2.0 / sample_count) * (prediction_batch - target_batch)


mean_squared_error = LossFunction(
    forward_fn=_mean_squared_error,
    derivative_fn=_mean_squared_error_grad,
)


def _cross_entropy(prediction_batch: FloatArray, target_batch: FloatArray) -> float:
    epsilon = 1e-12
    clipped_predictions = np.clip(prediction_batch, epsilon, 1.0 - epsilon)
    sample_losses = -np.sum(target_batch * np.log(clipped_predictions), axis=1)
    return float(np.mean(sample_losses))


def _cross_entropy_grad(prediction_batch: FloatArray, target_batch: FloatArray) -> FloatArray:
    sample_count = prediction_batch.shape[0]
    epsilon = 1e-12
    clipped_predictions = np.clip(prediction_batch, epsilon, 1.0 - epsilon)
    return (clipped_predictions - target_batch) / sample_count


cross_entropy = LossFunction(
    forward_fn=_cross_entropy,
    derivative_fn=_cross_entropy_grad,
)


__all__ = [
    "Loss",
    "LossFunction",
    "mean_squared_error",
    "cross_entropy",
]

