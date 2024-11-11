# Neural Network Implementation from Scratch

A simple implementation of a feedforward neural network with backpropagation using only NumPy. This implementation demonstrates the core concepts of neural networks including forward propagation, backpropagation, and gradient descent.

## Network Architecture
- Input Layer: 3 neurons
- Hidden Layer: 3 neurons with sigmoid activation
- Output Layer: 1 neuron (linear activation)
- Loss Function: Mean Squared Error (MSE)

## Features
- Forward propagation
- Backpropagation using chain rule
- Gradient descent optimization
- Parameter updates using learning rate

## Implementation Details
The network performs:
1. Forward pass to compute predictions
2. Loss calculation using MSE
3. Backward pass to compute gradients
4. Parameter updates using gradient descent

## Dependencies
```python
import numpy as np
