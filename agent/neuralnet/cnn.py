from nn import Layer  # pylint: disable=import-error
import numpy as np
import scipy.signal as sig


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, kernel_num, name):
        self.name = f"Conv-{name}"

        input_depth, input_height, input_width = input_shape

        self.kernel_num = kernel_num
        self.input_depth = input_depth
        self.input_shape = input_shape
        self.output_shape = (kernel_num, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (kernel_num, input_depth, kernel_size, kernel_size)
    
    def load_params(self, network_name):
        #Read from File
        # print(f"Loading {network_name}_{self.name}_weights.npy")
        self.kernels = np.load(f"{network_name}_{self.name}_weights.npy")
        # print(f"Loading {network_name}_{self.name}_biases.npy")
        self.biases = np.load(f"{network_name}_{self.name}_biases.npy")

    def randomize_params(self):
        # Randomly Initialize
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)

    def save_params(self, network_name):
        np.save( f"{network_name}_{self.name}_weights.npy", self.kernels)
        np.save( f"{network_name}_{self.name}_biases.npy", self.biases)

    def forward(self, layer_input):
        self.input = layer_input
        self.output = np.copy(self.biases)

        for i in range(self.kernel_num):
            for j in range(self.input_depth):
                self.output[i] += sig.correlate2d(self.input[j], self.kernels[i,j], "valid")
        return self.output

    def backward(self, d_outputs, learning_rate):
        d_kernels = np.zeros(self.kernel_shape)
        d_biases = d_outputs
        d_inputs = np.zeros(self.input_shape)

        for i in range(self.kernel_num):
            for j in range(self.input_depth):
                d_kernels[i, j] = sig.correlate2d(self.input[j], d_outputs[i], "valid")
                d_inputs[j] = sig.convolve2d(d_outputs[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * d_kernels
        self.biases -= learning_rate * d_biases

        return d_inputs


class Flatten(Layer):
    def __init__(self, input_shape, output_shape):
        self.name = "Flatten"
        self.input_shape = input_shape
        self.output_shape = output_shape

    def randomize_params(self):
        pass

    def load_params(self, network_name):
        pass

    def save_params(self, network_name):
        pass

    def forward(self, layer_input):
        return np.reshape(layer_input, self.output_shape)

    def backward(self, d_output, learning_rate):
        return np.reshape(d_output, self.input_shape)