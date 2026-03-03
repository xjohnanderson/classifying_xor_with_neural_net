 
# Core Neural Network logic including feedforward and backpropagation.
 
import numpy as np
from activations import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        #Initializes weights and biases for a 3-layer network. 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with a standard normal distribution
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Initialize biases to zero
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def feedforward(self, X):
        #Performs a forward pass through the network. 
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_activation)

        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = sigmoid(self.output_activation)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        #Performs backpropagation and updates weights/biases. 
        # Calculate output layer error and delta
        output_error = y - self.predicted_output
        output_delta = output_error * sigmoid_derivative(self.predicted_output)

        # Calculate hidden layer error and delta
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases using gradient descent
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        #Runs the training loop for a specified number of epochs. 
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 4000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")