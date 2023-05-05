import sys
from cnn import Convolutional, Flatten # pylint: disable=import-error
from nn import Layer, tanH, Sigmoid # pylint: disable=import-error

sys.path.append("neuralnet")
class AgentNetwork:
    """hyper_params : {is_randomized: Bool
        load_network: network name
    }
    """
    def __init__(self, hyper_params, network_name):
        
        self.shared = [
            Convolutional((14,7,7), 3, 20, "Shared1"),
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

        if (hyper_params["is_randomized"]):
            for layer in self.network:
                layer.randomize_params()

        else:
            #Load from files
            for layer in self.network:
                layer.load_params(hyper_params["load_network"])






