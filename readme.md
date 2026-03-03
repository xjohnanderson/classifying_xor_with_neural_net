Simple NumPy Neural Network

A lightweight, from-scratch implementation of a Feedforward Neural Network using NumPy. 

This project demonstrates the core mechanics of deep learning, including forward propagation, backpropagation, and gradient descent.

## Features
Manual Backpropagation: Implements the chain rule to update weights and biases.
Sigmoid Activation: Uses the sigmoid function for non-linearity.
Matrix Operations: Optimized using NumPy's vectorization for speed.
XOR Solver: Pre-configured to solve the non-linear XOR logic gate.

## Mathematical FoundationThe network relies on the following key components:
Feedforward: Calculations for each layer:$$z = X \cdot W + b$$$$a = \sigma(z)$$Activation (Sigmoid):$$\sigma(x) = \frac{1}{1 + e^{-x}}$$Loss Function: Mean Squared Error (MSE).

## Getting Started

### Prerequisites
Ensure you have Python installed along with the NumPy library:
Bashpip install numpy

### Usage
The NeuralNetwork class allows you to define the architecture (input, hidden, and output layers) dynamically.
Python
# Initialize: 2 input nodes, 4 hidden nodes, 1 output node
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Train the model
nn.train(X_train, y_train, epochs=10000, learning_rate=0.1)

# Predict
predictions = nn.feedforward(X_test)

## Training Progress
The model logs the Mean Squared Error (MSE) loss every 4,000 epochs to help you track convergence.
Epoch
Typical Loss (XOR)0~0.25+4000~0.01 - 0.058000< 0.005

## Project Structure__init__: Initializes weights with a standard normal distribution and biases to zero.feedforward: Passes input through the hidden layer to the output.backward: Calculates gradients and updates weights using the learning rate.train: The main loop coordinating the forward and backward passes.
