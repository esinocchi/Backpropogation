# Gradient Descent Implementation with Backpropagation

A Python implementation showing one complete step of gradient descent with backpropagation. This code demonstrates how neural networks learn by computing gradients and updating parameters.

## What This Code Demonstrates
- Forward propagation through two layers
- Computation of gradients using backpropagation
- One step of gradient descent parameter updates

## Structure
1. **Forward Pass**:
   - Input → Hidden Layer (with sigmoid activation)
   - Hidden Layer → Output
   - Loss Calculation

2. **Backward Pass (Backpropagation)**:
   - Computing gradients using chain rule
   - Starting from loss and working backwards
   - Computing gradients for all weights and biases

3. **Parameter Updates**:
   - Using gradient descent: θ_new = θ - α∇θ
   - Where α is learning rate and ∇θ is the gradient

## Mathematics
- **Forward**: z = wx + b
- **Activation**: sigmoid(z) = 1/(1 + e^(-z))
- **Loss**: L = (y - ŷ)²
- **Gradients**: Computed using chain rule
- Additional notes behind the mathematics can be found in [Math.pdf]

## Dependencies
```python
import numpy as np
