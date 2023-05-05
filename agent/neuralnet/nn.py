import numpy as np

class Layer:
    """This class encapsulates the logic for a dense feed-forward layer in a neural network
    """
    def __init__(self, input_size, neurons, name):
        self.name = f"FF-{name}"
        self.weights_shape = (neurons, input_size)
        self.bias_shape = (neurons, 1)

    def randomize_params(self):
        self.weights = np.random.randn(*self.weights_shape)
        self.bias = np.random.randn(*self.bias_shape)

    def load_params(self, network_name):
        self.weights = np.load(f"{network_name}_{self.name}_weights.npy")
        self.bias = np.load(f"{network_name}_{self.name}_biases.npy")

    def save_params(self, network_name):
        np.save( f"{network_name}_{self.name}_weights.npy", self.weights)
        np.save( f"{network_name}_{self.name}_biases.npy", self.bias)

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
    
    def randomize_params(self):
        pass

    def load_params(self, network_name):
        pass

    def save_params(self, network_name):
        pass

class ReLU(Activation):
    """This class outlines the ReLU activation function
    """
    def __init__(self):
        self.name = "ReLU"
        ReLU_function = lambda x : x * (x > 0)
        ReLU_prime = lambda x : 1 if np.greater(x, 0) else 0
        super().__init__(ReLU_function, ReLU_prime)


class tanH(Activation):
    def __init__(self):
        self.name = "tanH"
        tanH_function = lambda x : np.tanh(x)
        tanH_prime = lambda x : np.cosh(x) ** 2
        super().__init__(tanH_function, tanH_prime)

class Sigmoid(Activation):
    def __init__(self):
        self.name = "Sigmoid"
        sigmoid_function = lambda x : 1 / ( 1 + np.exp(-x))
        sigmoid_prime = lambda x : sigmoid_function(x) * (1 - sigmoid_function(x))
        super().__init__(sigmoid_function, sigmoid_prime)