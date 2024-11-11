import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1-s)

def Net(): # 3 input 1 output
    # Initialize parameters
    x = np.array([2, 3, 1])  # inputs
    w1 = np.array([[0.5, 0.3, -0.1],    # weights for first hidden layer
                   [0.2, 0.4, 0.6],     
                   [-0.3, 0.5, 0.2]])   
    b1 = np.array([0.1, 0.2, 0.3])      # biases for hidden layer
    w2 = np.array([0.8, -0.5, 0.4])     # weights for output layer
    b2 = 0.3                            # bias for output layer
    actual = 1                          # true value

    print("Forward Pass")
    print("-----------")

    # Forward Pass
    z1 = np.dot(w1, x) + b1            # hidden layer pre-activation
    h = sigmoid(z1)                     # hidden layer activation
    z2 = np.dot(w2, h) + b2            # output layer pre-activation
    predicted = z2
    loss = (actual - predicted)**2

    print(f"Predicted: {predicted:.4f}")
    print(f"Actual: {actual}")
    print(f"Loss: {loss:.4f}")

    print("\nBackpropagation")
    print("--------------")

    # Backpropagation
    # 1. Output layer gradients
    dL_dpred = -2 * (actual - predicted)
    dL_dw2 = dL_dpred * h
    dL_db2 = dL_dpred

    # 2. Hidden layer gradients
    dL_dh = dL_dpred * w2
    dh_dz1 = sigmoid_derivative(z1)
    dL_dz1 = dL_dh * dh_dz1
    dL_dw1 = np.outer(dL_dz1, x)
    dL_db1 = dL_dz1

    # Parameter updates
    learning_rate = 0.01
    w1_new = w1 - learning_rate * dL_dw1
    b1_new = b1 - learning_rate * dL_db1
    w2_new = w2 - learning_rate * dL_dw2
    b2_new = b2 - learning_rate * dL_db2

    print("\nParameter Updates:")
    print(f"Loss: {loss:.4f} -> {(actual - np.dot(w2_new, sigmoid(np.dot(w1_new, x) + b1_new)) - b2_new)**2:.4f}")

Net()
