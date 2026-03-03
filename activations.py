 
# Mathematical activation functions and their derivatives.
 
import numpy as np

def sigmoid(x):
    #Computes the sigmoid activation function. 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    #Computes the derivative of the sigmoid function based on its output. 
    return x * (1 - x)