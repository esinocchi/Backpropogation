import numpy as np
import torch
import torch.nn as nn
# Showing calculations of forward pass, gradient descent and back propogation

def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

def sigmoid_derivative(x):
    s = sigmoid(x)
    d = s * (1-s)
    return d

def Net(): # 3 input 1 output
    x = np.array([2, 3, 1]) # inputs
    w1 = np.array([[0.5, 0.3, -0.1],    # weights for first hidden neuron
                   [0.2, 0.4, 0.6],     # weights for second hidden neuron
                   [-0.3, 0.5, 0.2]])   # weights for third hidden neuron
    
    b1 = np.array([0.1, 0.2, 0.3])      # biases for hidden layer (one per neuron)
    
    w2 = np.array([0.8, -0.5, 0.4])     # weights for output neuron (one per hidden neuron)
    b2 = 0.3                            # bias for output neuron
    actual = 1

    print("Forward Pass")
    print("-----------")

    # Forward Pass

    # First layer
    z1 = np.dot(w1, x) + b1 # weighted sums + biases for each hidden neuron
    h = sigmoid(z1) # hidden layer activation

    # Second layer
    z2 = np.dot(w2, h) + b2 # weighted sums + biases for output
    predicted = z2

    loss = (actual - predicted)**2




    print(f"Inputs (x): {x}")
    
    print(f"\nFirst Layer (Hidden Layer):")
    print("Each hidden neuron's calculation:")
    for i in range(3):
        print(f"\nHidden Neuron {i+1}:")
        print(f"Weights: {w1[i]}")
        print(f"Weighted sum = z1_{i+1} = ({w1[i][0]}*{x[0]}) + ({w1[i][1]}*{x[1]}) + ({w1[i][2]}*{x[2]}) + {b1[i]} = {z1[i]:.1f}")
        print(f"h_{i+1} = sigmoid(z1_{i+1}) = {h[i]:.4f}")

    print("\nSummary of Hidden Layer 1:\n")
    print(f"z1 = {z1}")
    print(f"h = {h} (After Sigmoid Operation)\n")
    
    print(f"\nSecond Layer (Output Layer):")
    print("Output neuron calculation:")
    print(f"Weights: {w2}")
    print(f"Output = ({w2[0]}x{h[0]:.4f}) + ({w2[1]}x{h[1]:.4f}) + ({w2[2]}x{h[2]:.4f}) + {b2}")
    print(f"predicted = {predicted:.4f}")
    print(f"Actual (y): {actual}")
    print(f"Loss: {loss:.4f}")

    # Backward Pass

    print("\nBackpropagation (Computing Gradients)")
    print("-----------")

    # Step 1: Calculate dL/dpred (starting from loss)
    dL_dpred = -2 * (actual - predicted)
    print(f"\nStep 1: dL/dy_hat = -2(y - y_hat) = {dL_dpred:.4f}")
    
    # Step 2: Calculate gradients for output layer (w2, b2)
    dL_dw2 = dL_dpred * h          # dL/dw2 = (dL/dy_hat)(h)
    dL_db2 = dL_dpred              # dL/db2 = dL/dy_hat
    
    print(f"\nStep 2: Output Layer Gradients:")
    print(f"dL/dw2 = (dL/dy_hat)h:")
    for i in range(len(w2)):
        print(f"For w2_{i+1}: {dL_dpred:.4f} * {h[i]:.4f} = {dL_dw2[i]:.4f}")
    print(f"dL/db2 = {dL_db2:.4f}")
    
    # Step 3: Calculate dL/dh
    dL_dh = dL_dpred * w2          # dL/dh = (dL/dy_hat)(w2)
    print(f"\nStep 3: dL/dh = (dL/dy_hat)w2:")
    for i in range(len(h)):
        print(f"For h_{i+1}: {dL_dpred:.4f} * {w2[i]:.4f} = {dL_dh[i]:.4f}")
    
    # Step 4: Calculate dL/dz1
    dh_dz1 = sigmoid_derivative(z1)  # dh/dz1
    dL_dz1 = dL_dh * dh_dz1         # dL/dz1 = (dL/dh)(dh/dz1)
    
    print(f"\nStep 4: dL/dz1 = (dL/dh)(dh/dz1):")
    for i in range(len(z1)):
        print(f"For z1_{i+1}: {dL_dh[i]:.4f} * {dh_dz1[i]:.4f} = {dL_dz1[i]:.4f}")
    
    # Step 5: Calculate gradients for hidden layer (w1, b1)
    dL_dw1 = np.outer(dL_dz1, x)    # dL/dw1 = (dL/dz1)(x)
    dL_db1 = dL_dz1                 # dL/db1 = dL/dz1
    
    print(f"\nStep 5: Hidden Layer Gradients:")
    print("dL/dw1:")
    print(dL_dw1)
    print("\ndL/db1:")
    print(dL_db1)
    
    # Step 6: Update all parameters
    learning_rate = 0.01
    
    w1_new = w1 - learning_rate * dL_dw1
    b1_new = b1 - learning_rate * dL_db1
    w2_new = w2 - learning_rate * dL_dw2
    b2_new = b2 - learning_rate * dL_db2
    
    print("\nParameter Updates (learning_rate = 0.1):")
    print("First Layer:")
    print(f"w1: \n{w1}\n -> \n{w1_new}")
    print(f"b1: {b1} -> {b1_new}")
    print("\nSecond Layer:")
    print(f"w2: {w2} -> {w2_new}")
    print(f"b2: {b2:.4f} -> {b2_new:.4f}")

Net()