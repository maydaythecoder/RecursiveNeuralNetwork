# RecursiveNeuralNetwork

## Overview

RecursiveNeuralNetwork is a NumPy-only sandbox for experimenting with fully-connected neural networks from first principles. It exposes composable modules for activations, losses, layers, and optimizers so you can trace every tensor flowing through the system without relying on an automatic differentiation framework.

## Key Features

- **Typed NumPy primitives** for activations, losses, and optimizers.
- **Modular layer graph** built from reusable `Node`, `InputLayer`, `HiddenLayer`, and `OutputLayer` classes.
- **Beginner-friendly naming** (`input_batch`, `weight_matrix`, `parameter_pairs`, etc.) so the forward/backward flow reads like plain English.
- **Training utilities** including a gradient checker and a self-contained demo on a synthetic dataset.
- **Documentation** detailing the architecture and reasoning behind each component.

## Environment Setup

1. Use Python 3.11 or later to get full support for the type annotations used in the project.
2. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)
3. Install dependencies:
   - `pip install -r requirements.txt`
4. (Optional) Install type stubs so Pyright/BasedPyright can resolve NumPy types:
   - `pip install types-numpy`

## Usage

### Quick Start

Verify your environment and watch the network train on the synthetic dataset:

```bash
cd /Users/muhyadinmohamed/Documents/Development/Python/RecursiveNeuralNetwork
source venv/bin/activate
python -m structure.network
```

The module prints the gradient check score, epoch-level training losses, and final accuracy. Use this as a template when wiring custom datasets or experimenting with deeper networks.

## Project Structure

``` txt
logic/
  activationFunction.py   # Activation primitives and derivatives
  lossFunction.py         # Loss functions and gradients
  Optimizer.py            # Optimizers (currently SGD with clipping)
structure/
  InputLayer.py           # Input validation and input_batch passthrough cache
  Node.py                 # Parameter container (`weight_matrix`/`bias_vector`) with He initialization
  hiddenLayer.py          # Dense hidden layer with activation and optimizer-friendly parameter_pairs
  outputLayer.py          # Output layer, prediction utilities, and activation-derivative toggle
  network.py              # Network assembly, gradient check, and demo training loop
docs/                     # Architecture and component guides
requirements.txt          # Runtime dependencies
```

## Documentation

Additional technical notes live in the `docs/` directory:

- `docs/architecture.md` — network topology, data flow, and gradient propagation.
- `docs/modules.md` — module-by-module design rationale with external references.
- `docs/training-guide.md` — training utilities, gradient checking workflow, and extension tips.

Refer to those guides for deeper dives into the implementation, design trade-offs, and links to authoritative resources that inspired the project.

## Contributing

Contributions are welcome. Open an issue or submit a pull request if you would like to:

- Extend the optimizer set (e.g., Momentum, Adam)
- Add alternative activation or loss functions
- Port the training loop to additional datasets
- Improve the documentation or examples
