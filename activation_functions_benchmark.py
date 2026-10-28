"""
Deep Learning Activation Functions Benchmark & Analysis
Author: Akash V (@akashvenkey)
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Numerically stable Sigmoid activation."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """Rectified Linear Unit (ReLU) activation."""
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def tanh(x):
    """Hyperbolic Tangent activation."""
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

def run_activation_benchmark():
    x = np.linspace(-5.0, 5.0, 200)
    print("=== Neural Activation Functions Benchmark ===")
    print(f"Sample test vector (x): [-3, -1, 0, 1, 3]")
    sample = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print(f"Sigmoid: {np.round(sigmoid(sample), 4)}")
    print(f"ReLU:    {np.round(relu(sample), 4)}")
    print(f"Tanh:    {np.round(tanh(sample), 4)}")

if __name__ == "__main__":
    run_activation_benchmark()