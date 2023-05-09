import sys
sys.path.append("neuralnet")
from neuralnet.cnn import Convolutional, Flatten # pylint: disable=import-error
from neuralnet.nn import Layer, tanH, Sigmoid # pylint: disable=import-error
from neuralnet.model_IO import save_model, load_model # pylint: disable=import-error



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
    
    def train(self, examples, policy_targets, value_targets, learning_rate):
        """
        Trains the network using batch gradient descent.
        `examples`: List of examples (numpy arrays) used for training
        `policy_targets`: List of policy targets (numpy arrays) for each state
        `value_targets`: List of value targets (numpy arrays) for each state
        `learning_rate`: Learning rate for gradient descent
        """
        batch_size = len(examples)
        policy_grads = []
        value_grads = []
        
        # Compute gradients for each sample in the batch
        for i in range(batch_size):
            state = examples[i][0]
            policy_target = policy_targets[i]
            value_target = examples[i][2]
            
            shared_output = self.process_shared(state)
            policy_output = self.get_policy(state)
            value_output = self.get_value(state)
            
            # Compute gradients for policy and value outputs
            policy_loss_grad = (policy_output - policy_target) / batch_size
            value_loss_grad = (value_output - value_target) / batch_size
            
            # Backpropagate through value layers
            d_output = value_loss_grad * self.value
    
    def save_network(self):
        save_model(self)