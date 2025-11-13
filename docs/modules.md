# Module Reference

## `logic/activationFunction.py`

- Defines the `UnaryActivation` dataclass pairing `forward_fn` and `derivative_fn` callables for clarity.
- Ships with `sigmoid`, `relu`, and `softmax` implementations using numerically stable NumPy operations (e.g., clipping to mitigate overflow).  
- Further reading: [NumPy Universal Functions](https://numpy.org/doc/stable/reference/ufuncs.html), [Softmax Stabilization Techniques](https://peterroelants.github.io/posts/neural-networks-softmax/).

## `logic/lossFunction.py`

- Provides the `LossFunction` wrapper so losses expose consistent `forward_fn`/`derivative_fn` signatures.
- Includes mean squared error for regression and cross-entropy for multi-class classification.  
- References: [Deep Learning Book, Chapter 6.2](https://www.deeplearningbook.org/contents/mlp.html), [Cross-Entropy Loss Explained](https://machinelearningmastery.com/cross-entropy-for-machine-learning/).

## `logic/Optimizer.py`

- Implements stochastic gradient descent with optional gradient clipping via `np.linalg.norm`.
- Design encourages future optimizers to adhere to the `Optimizer` protocol and consume iterables of `ParameterGradientPair`.  
- Additional material: [SGD Overview](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf), [Gradient Clipping in Practice](https://paperswithcode.com/method/gradient-clipping).

## `structure/Node.py`

- Encapsulates `weight_matrix`/`bias_vector` tensors with matching gradient buffers and exposes He initialization.
- Keeps parameter tensors centralized, simplifying optimizer integration.  
- Background: [He Initialization](https://arxiv.org/abs/1502.01852).

## `structure/InputLayer.py`

- Validates input dimensionality and forwards tensors downstream unchanged.
- Serves as an explicit entry point for future preprocessing hooks.  
- Related reading: [Input Normalization Strategies](https://developers.google.com/machine-learning/crash-course/representation/input-features) for potential extensions.

## `structure/hiddenLayer.py`

- Performs dense affine transforms followed by configurable activations.
- Caches `cached_input_batch`/`cached_activation` tensors for gradient computation and returns `parameter_pairs` for the optimizer.  
- Further reading: [Backpropagation Mechanics](https://cs231n.github.io/optimization-2/).

## `structure/outputLayer.py`

- Mirrors hidden layer behavior but includes prediction helpers and an activation-derivative toggle for loss-specific optimizations.
- Default activation is softmax, suitable for classification problems, and it exposes `parameter_pairs` alongside cached tensors for training.  
- Reference: [Softmax + Cross-Entropy Gradient Derivation](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/).

## `structure/network.py`

- Aggregates layers into a `NeuralNetwork` class, providing training, evaluation, and gradient check utilities.
- Houses dataset generation (`make_toy_classification_dataset`) and a demo entry point.  
- Suggested reading: [Gradient Checking](https://cs231n.github.io/neural-networks-3/#gradcheck) and [Two-Moons Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).
