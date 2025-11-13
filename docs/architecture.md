# Architecture Guide

## Data Flow

1. **Input ingestion** — `InputLayer` validates feature dimensionality and forwards raw tensors downstream without modification.
2. **Hidden transformations** — each `HiddenLayer` performs an affine transformation (`X @ W + b`) followed by a non-linear activation supplied through `UnaryActivation`. Intermediate tensors are cached to support backpropagation.
3. **Output projection** — `OutputLayer` produces logits that are converted to probabilities by its activation (softmax by default) and exposes prediction helpers for evaluation.

## Parameter Management

- `Node` encapsulates `weight_matrix`/`bias_vector` tensors alongside gradient buffers. Opting for a dedicated container decouples layer logic from optimizer state and lowers the risk of mismatched shapes.
- Weights use **He initialization** to maintain activation variance through ReLU-heavy networks. See the original paper for details: [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852).

## Gradient Propagation

- Backprop in `HiddenLayer` and `OutputLayer` follows the chain rule: multiply the upstream gradient by the activation derivative, accumulate batch-averaged parameter gradients, and propagate downstream gradients using the transpose of the cached `weight_matrix`.
- `OutputLayer.backward` includes an `apply_activation_derivative` flag so cross-entropy with softmax can leverage the simplified gradient (`y_hat - y`). For the derivation, refer to [Stanford CS231n Notes](https://cs231n.github.io/linear-classify/#softmax).

## Optimization Loop

- `NeuralNetwork.train_step` sequences forward pass → loss evaluation → backward pass → optimizer update → gradient reset. This mirrors modern deep learning frameworks while remaining explicit enough to trace the math.
- `SGD` optionally clips gradients to prevent exploding updates. For a quick refresher on clipping strategies, see [Deep Learning Book, Chapter 6.5](https://www.deeplearningbook.org/contents/optimization.html).

## Utilities

- `gradient_check` compares analytic and finite-difference gradients to flag implementation mistakes. It follows the procedure described in [Andrew Ng’s Coursera notes](https://cs229.stanford.edu/notes2020spring/cs229-notes-backprop.pdf).
- `make_toy_classification_dataset` generates a two-moons-style dataset inspired by [scikit-learn’s datasets.make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) but keeps dependencies limited to NumPy.

## Extensibility Considerations

- New layers can conform to the same `forward/backward/parameter_pairs` interface, allowing plug-and-play experimentation.
- Switching to different losses or activations requires only swapping imports in `structure/network.py`, thanks to the `LossFunction` and `UnaryActivation` abstractions.
