 
#Entry point for training and evaluating the Neural Network.
 
import numpy as np
from network import NeuralNetwork

def run_xor_example():
    # Dataset: XOR Problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize Model
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # Train
    print("Starting training...")
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Inference
    output = nn.feedforward(X)
    print("\nPredictions after training:")
    print(output)

if __name__ == "__main__":
    run_xor_example()