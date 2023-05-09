import sys
sys.path.append("neuralnet")
from sgd import StochasticGradientDescent # pylint: disable=import-error
from neuralnet.cnn import Convolutional, Flatten # pylint: disable=import-error
from neuralnet.nn import Layer, tanH, Sigmoid # pylint: disable=import-error
from neuralnet.model_IO import save_model, load_model # pylint: disable=import-error
import numpy as np
import torch


class AgentNetwork:
    """hyper_params =  {is_randomized: Bool,
        load_network: string name of the network to load,
        input_depth: Depth of the input sample
    }
    """
    def __init__(self, hyper_params, network_name):
        self.network_name = network_name
        self.input_depth = hyper_params["input_depth"]
        self.optimizer = StochasticGradientDescent()
        
        # Create network
        self.shared = [
            Convolutional((self.input_depth,7,7), 3, 20, "Shared1"),
            tanH(),
            Convolutional((20,5,5), 3, 20, "Shared2"),
            tanH(),
            Flatten((20,3,3), (180,1)),
            Layer(180, 100, "Shared3"),
            Sigmoid()
        ]
        self.policy_layers = [
            Layer(100, 343, "Policy1"),
            Sigmoid()
        ]

        self.value_layers = [
            Layer(100, 1, "Value1"),
            tanH()
        ]

        self.network = [*self.shared, *self.policy_layers, *self.value_layers]

        # Initialize params in each layer
        for layer in self.network:
            layer.randomize_params()
        
        if (hyper_params["is_randomized"] is False):
            #Load from files
            load_model(self, hyper_params["load_network"])

    def _process_shared(self, input_state):
        if input_state.shape != (self.input_depth, 7, 7):
            raise ValueError("Input shape is not what is defined in hyper_param")

        output = input_state
        for layer in self.shared:
            output = layer.forward(output)
        return output

    def get_policy(self, input_state):
        shared_output = self._process_shared(input_state)
        
        output = shared_output
        for layer in self.policy_layers:
            output = layer.forward(output)

        return output
        
    
    def get_value(self, input_state):
        shared_output = self._process_shared(input_state)
        
        output = shared_output
        for layer in self.value_layers:
            output = layer.forward(output)

        return output


    def backward(self, d_out):
        """This method sets the gradients of the params in each layer
        of the network

        Args:
            d_out: The gradient of loss w.r.t output
        """
        d_out_back = d_out
        for layer in reversed(self.network):
            d_out_back = layer.backwards(d_out_back)

    def train(self, training_examples):
        for state, improved_policy, value in training_examples:

            # Compute the predicted policy and value
            predicted_policy, predicted_value = self.get_policy(state), self.get_value(state)
                
            # Compute the total loss
            total_loss = self.optimizer.loss_function(predicted_value, value,\
                                                      predicted_policy, improved_policy)
                
            # Compute the gradient of the output w.r.t total loss
            d_out_policy = -np.divide(improved_policy, predicted_policy)
            d_out_value = -2 * (value - predicted_value)

            # Backpropagate the output gradient through the network
            self.backward(d_out_policy + d_out_value)

            # Update the network parameters using the SGD optimizer
            for layer in self.network:
                params = layer.get_params()
                if params is None:
                    continue

                weights, biases = params
                d_weights, d_biases = layer.get_grads()

                self.optimizer.update_params(weights, d_weights)
                self.optimizer.update_params(biases, d_biases)

            # Zero out the gradients for the next iteration
            for layer in self.network:
                layer.zero_grad()
        

    def save_network(self):
        save_model(self)
        