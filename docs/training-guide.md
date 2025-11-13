# Training Guide

## Running the Demo

Execute the built-in training script to see the entire pipeline:

``` bash
cd /Users/muhyadinmohamed/Documents/Development/Python/RecursiveNeuralNetwork
source venv/bin/activate
python -m structure.network
```

- Generates a synthetic two-class dataset.
- Runs a 2x16 hidden-layer MLP for 50 epochs with mini-batch SGD.
- Prints gradient-check diagnostics, loss progression, and final accuracy.

## Gradient Checking Workflow

1. Slice a small batch (e.g., 5 samples) to keep finite-difference computation manageable.
2. Call `gradient_check(network, inputs, targets)` to compare analytic and numerical gradients.
3. Investigate any discrepancies > 1e-6. Common culprits include shape broadcasting mistakes or missing activation derivatives.  
   - Reference: [CS229 Backprop Notes](https://cs229.stanford.edu/notes2020spring/cs229-notes-backprop.pdf)

## Hyperparameter Adjustments

- **Learning rate** — tune via the `SGD` constructor. Start around `0.1` and scale down if the loss diverges.  
  - See [Practical Tips for SGD](https://arxiv.org/abs/1206.5533).
- **Gradient clipping** — tighten `max_grad_norm` when experimenting with deeper networks to curb exploding gradients.
- **Hidden widths/depths** — modify the `hidden_units` tuple in `build_demo_network`. You can combine ELU, GELU, or other activations by extending `logic/activationFunction.py`.

## Extending to Real Datasets

- Replace `make_toy_classification_dataset` with your data loader. Ensure inputs are `float64` NumPy arrays and targets are one-hot encoded when using softmax + cross-entropy.
- For regression, swap in `mean_squared_error` and configure `OutputLayer` with an identity activation (e.g., add a lambda returning inputs unchanged).  
  - Reference: [One-Hot Encoding with NumPy](https://numpy.org/doc/stable/reference/generated/numpy.eye.html).

## Debugging Tips

- Monitor norms of weights and gradients each epoch to catch instability early. NumPy’s `np.linalg.norm` is handy for quick checks.
- Visualize the decision boundary by projecting predictions onto a grid; see [Matplotlib Contour Plots](https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html) for inspiration.
- When experimenting with new activations, verify their derivatives numerically before plugging them into `HiddenLayer`.

## Suggested Next Steps

- Implement additional optimizers (Momentum, RMSProp, Adam) for faster convergence; consult the [Optimization Algorithms Cheatsheet](https://ruder.io/optimizing-gradient-descent/).
- Add dropout or batch normalization to explore regularization techniques. Resources: [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html) and [BatchNorm](https://arxiv.org/abs/1502.03167).
