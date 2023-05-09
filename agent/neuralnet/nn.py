import numpy as np

class Layer:
    """This class encapsulates the logic for a dense feed-forward layer in a neural network
    """
    def __init__(self, input_size, neurons, name):
        self.name = f"FF-{name}"
        self.weights_shape = (neurons, input_size)
        self.biases_shape = (neurons, 1)

    def randomize_params(self):
        self.weights = np.random.randn(*self.weights_shape)
        self.biases = np.random.randn(*self.biases_shape)

    def forward(self, layer_input):
        self.input = layer_input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, d_output):
        self.d_weights = np.dot(d_output, self.input.T)
        self.d_biases = d_output
        
        return np.dot(self.weights.T, d_output)
    
    def get_params(self):
        return (self.weights, self.biases)
    
    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def get_grads(self):
        return (self.d_weights, self.d_biases)

    def zero_grad(self):
        self.d_weights = np.zeros(self.weights_shape)
        self.d_biases = np.zeros(self.biases_shape)

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

    def get_params(self):
        return None
    
    def set_params(self, weights, biases):
        pass

    def zero_grad(self):
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