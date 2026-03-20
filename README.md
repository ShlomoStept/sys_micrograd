# sys_micrograd: Autograd Engine from Scratch

A scalar-valued automatic differentiation engine and neural network library, built from scratch in Python. Implements reverse-mode autodiff (backpropagation) with a PyTorch-like API for educational purposes.

## Overview

Every deep learning framework relies on automatic differentiation to compute gradients. This project implements that core mechanism from first principles: a `Value` class that wraps scalar data, tracks operations in a directed acyclic computation graph, and propagates gradients backward through the chain rule.

On top of the autograd engine, a minimal neural network library provides `Neuron`, `Layer`, and `MLP` classes that can be composed, trained with gradient descent, and visualized.

## The Autograd Engine

The `Value` class (`micrograd_engine.py`) is the foundation. Each `Value` wraps:

- **`data`**: the scalar value
- **`grad`**: the gradient (accumulated during backpropagation)
- **`_backward`**: a closure that computes the local gradient contribution
- **`_prev`**: the set of parent `Value` nodes in the computation graph

### Supported Operations

| Operation | Forward | Backward (local gradient) |
|-----------|---------|---------------------------|
| `a + b` | `a.data + b.data` | `da += 1.0 * dout`, `db += 1.0 * dout` |
| `a * b` | `a.data * b.data` | `da += b.data * dout`, `db += a.data * dout` |
| `a ** n` | `a.data ** n` | `da += n * a.data^(n-1) * dout` |
| `a.exp()` | `e^(a.data)` | `da += e^(a.data) * dout` |
| `a.tanh()` | `tanh(a.data)` | `da += (1 - t^2) * dout` |
| `a - b` | via `a + (-b)` | derived from add and negate |
| `a / b` | via `a * b^(-1)` | derived from mul and pow |

All backward functions **accumulate** gradients (`+=`), which is essential for correct gradient computation when a value is used multiple times in the graph.

### Backpropagation

The `backward()` method performs a topological sort of the computation graph, then walks it in reverse order, calling each node's `_backward` closure:

```python
loss = ...  # some computation
loss.backward()  # computes gradients for all upstream Values
```

This implements the same algorithm as PyTorch's `loss.backward()`, but on scalars instead of tensors.

## Neural Network Library

Built on top of the autograd engine (`neural_network.py`):

```python
from sys_micrograd.neural_network import MLP

# 3 inputs -> hidden layer of 4 -> hidden layer of 4 -> 1 output
model = MLP(3, [4, 4, 1])

# Training data
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

# Training loop
for epoch in range(20):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yp - yt)**2 for yt, yp in zip(ys, ypred))

    # Backward pass (CRITICAL: zero gradients first)
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # Update weights
    for p in model.parameters():
        p.data += -0.05 * p.grad

    print(f"Epoch {epoch}, Loss = {loss.data}")
```

### Architecture

- **`Neuron`**: Stores weights and bias as `Value` objects. Forward pass computes `tanh(sum(w*x) + b)`.
- **`Layer`**: A list of `Neuron` objects. Each neuron receives the same input and produces one output.
- **`MLP`**: A list of `Layer` objects chained sequentially.

## Computation Graph Visualization

The `visualize_network.py` module generates Graphviz diagrams of the computation graph, showing data values, gradients, and operations at each node:

```python
from sys_micrograd.visualize_network import draw_dot

# After building a computation graph and calling backward()
dot = draw_dot(loss)
dot.render('graph', format='svg')
```

Each node displays `{label | data | grad}` and operation nodes show `+`, `*`, `tanh`, etc.

## The Gradient Bug Fix Story

A common and instructive bug in autograd implementations: **forgetting to zero gradients between training steps**. Without `p.grad = 0.0` before each `loss.backward()`, gradients accumulate across iterations. The effect is subtle: training appears to work but takes enormous, erratic steps because each backward pass adds to the previous iteration's gradients. The loss oscillates wildly instead of converging smoothly.

This is the same issue that PyTorch addresses with `optimizer.zero_grad()` -- this implementation makes the necessity visceral because you see the raw gradient values.

## Project Structure

```
sys_micrograd/
├── micrograd_engine.py      # Value class: autograd engine with backward()
├── neural_network.py        # Neuron, Layer, MLP: neural network building blocks
├── visualize_network.py     # Graphviz computation graph visualization
├── test1.py                 # Training demo: MLP on a 4-example classification task
└── __init__.py
```

## Requirements

- Python 3.7+
- Graphviz (`pip install graphviz`) -- for computation graph visualization
- NumPy, Matplotlib -- for the test script

## Project Context

Built as an educational deep dive into how automatic differentiation works under the hood. Follows the pedagogical approach of Andrej Karpathy's micrograd, with every component implemented and reasoned through independently.

This pairs with a three-part blog series covering:
1. The `Value` class and manual backpropagation
2. Automatic backpropagation via topological sort
3. Building and training the neural network

## License

MIT
