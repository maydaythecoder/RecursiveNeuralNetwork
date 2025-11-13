from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


FloatArray = NDArray[np.float64]


@dataclass
class InputLayer:
    """
    No learnable parameters; validates incoming feature dimensions.

    Forward simply caches the batch for downstream layers. Backward passes the gradient upstream
    unchanged.
    """

    input_dim: int
    cached_input_batch: FloatArray | None = None

    def forward(self, input_batch: FloatArray) -> FloatArray:
        if input_batch.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, received {input_batch.shape[1]}"
            )
        self.cached_input_batch = input_batch
        return input_batch

    def backward(self, output_gradient: FloatArray) -> FloatArray:
        # SAFETY: Input layer has no parameters, so upstream gradient flows through untouched.
        return output_gradient


__all__ = ["InputLayer"]

