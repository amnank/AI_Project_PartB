import sys
sys.path.append("neuralnet")
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

    def process_shared(self, input_state):
        if input_state.shape != (self.input_depth, 7, 7):
            raise ValueError("Input shape is not what is defined in hyper_param")

        output = input_state
        for layer in self.shared:
            output = layer.forward(output)
        return output

    def get_policy(self, input_state):
        shared_output = self.process_shared(input_state)
        
        output = shared_output
        for layer in self.policy_layers:
            output = layer.forward(output)

        return output
        
    
    def get_value(self, input_state):
        shared_output = self.process_shared(input_state)
        
        output = shared_output
        for layer in self.value_layers:
            output = layer.forward(output)

        return output
    
    
    def get_params(self):
        params = []
        for layer in self.network:
            params.append(layer.get_params())
        return params


    def perform_sgd(self, d_weights):
        learning_rate = 0.001
        for layer in self.agent_network.layers:
            if isinstance(layer, Layer):
                layer.weights -= learning_rate * layer.d_weights
            elif isinstance(layer, Convolutional):
                layer.weights -= learning_rate * layer.d_weights


    def backward(self, examples):
        pass

    def train(self, training_examples):
        for state, action_played, improved_policy, value in training_examples:

            # Compute the predicted policy and value
            predicted_policy, predicted_value = self.get_policy, self.get_value
                
            # Compute the policy loss and value loss
            policy_loss = -improved_policy * np.log(predicted_policy)
            value_loss = (value - predicted_value) ** 2
                
            # Compute the total loss
            total_loss = policy_loss + value_loss
                
            # Compute the gradient (delta) with respect to the total loss
            delta = np.gradient(total_loss, self.parameters())
                
            # Backpropagate the gradient (delta) through the network
            self.backward(delta)

            # Update the network parameters using the SGD optimizer
            self.perform_sgd(delta)

            # Zero out the gradient for the next iteration
            for layer in self.layers:
                layer.weights_grad = np.zeros_like(layer.weights)
        

    def save_network(self):
        save_model(self)


        