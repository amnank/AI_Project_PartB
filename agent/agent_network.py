import sys
import os
sys.path.append("neuralnet")
from cnn import Convolutional, Flatten # pylint: disable=import-error
from nn import Layer, tanH, Sigmoid # pylint: disable=import-error



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
        if (hyper_params["is_randomized"]):
            for layer in self.network:
                layer.randomize_params()

        else:
            #Load from files
            for layer in self.network:
                layer.load_params(hyper_params["load_network"])

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
    
    def save_params(self):
        if not os.path.exists(self.network_name):
            os.makedirs(self.network_name)

        for layer in self.network:
                os.path.join(self.network_name, )
                layer.save_params(self.network_name)