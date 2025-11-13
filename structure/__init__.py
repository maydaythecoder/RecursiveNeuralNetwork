from importlib import import_module
from typing import Any

from structure.InputLayer import InputLayer
from structure.Node import Node
from structure.hiddenLayer import HiddenLayer
from structure.outputLayer import OutputLayer

__all__ = [
    "InputLayer",
    "Node",
    "HiddenLayer",
    "OutputLayer",
    "NeuralNetwork",
    "build_demo_network",
    "demo_training_epoch",
    "gradient_check",
    "make_toy_classification_dataset",
]

_NETWORK_EXPORTS = {
    "NeuralNetwork",
    "build_demo_network",
    "demo_training_epoch",
    "gradient_check",
    "make_toy_classification_dataset",
}


def __getattr__(attribute: str) -> Any:
    if attribute in _NETWORK_EXPORTS:
        module = import_module("structure.network")
        return getattr(module, attribute)
    raise AttributeError(f"module 'structure' has no attribute '{attribute}'")

