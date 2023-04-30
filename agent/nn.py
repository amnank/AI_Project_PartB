import numpy as np

class Layer:
    """This class encapsulates the logic for a dense feed-forward layer in a neural network
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, layer_input):
        self.input = layer_input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, d_output, learning_rate):
        d_weights = np.dot(d_output, self.input.T)
        d_bias = d_output

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        return np.dot(self.weights.T, d_output)

class Activation(Layer):
    """This class encapsulates the logic of an activation layer that applies an activation function
    """
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, layer_input):
        self.input = layer_input
        return self.activation(self.input)

    def backward(self, d_output, learning_rate):
        return np.multiply(d_output, self.activation_prime(self.input))

class ReLU(Activation):
    """This class outlines the ReLU activation function
    """
    def __init__(self):
        ReLU_function = lambda x : x if x > 0 else 0
        ReLU_prime = lambda x : 1 if x > 0 else 0
        super().__init__(ReLU, ReLU_prime)

