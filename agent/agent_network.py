import sys
sys.path.append("neuralnet")
from .sgd import StochasticGradientDescent, Adam # pylint: disable=import-error
from .neuralnet.cnn import Convolutional, Flatten # pylint: disable=import-error
from .neuralnet.nn import Layer, tanH, Sigmoid # pylint: disable=import-error
from .neuralnet.model_IO import save_model, load_model # pylint: disable=import-error
import numpy as np
from .alpha_zero_helper import normalize_policy # pylint: disable=import-error


class AgentNetwork:
    """hyper_params =  {is_randomized: Bool,
        load_network: string name of the network to load,
        input_depth: Depth of the input sample
    }
    """
    def __init__(self, hyper_params, network_name):
        self.network_name = network_name
        self.input_depth = hyper_params["input_depth"]
        #self.optimizer = StochasticGradientDescent()
        self.optimizer = Adam()
        
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
            print(input_state.shape)
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


    def backward_shared(self, d_out):
        """This method sets the gradients of the params in each layer
        of the network

        Args:
            d_out: The gradient of loss w.r.t output
        """
        
        d_out_shared = d_out
        for layer in reversed(self.shared):
            d_out_shared = layer.backward(d_out_shared, self.optimizer.l2_lambda * 2)

    
    def backward_policy(self, d_out):
        """This method sets the gradients of the params in each layer
        of the network

        Args:
            d_out: The gradient of loss w.r.t output
        """
        
        d_out_policy = d_out
        for layer in reversed(self.policy_layers):
            d_out_policy = layer.backward(d_out_policy, self.optimizer.l2_lambda * 2)

        return d_out_policy
    
    def backward_value(self, d_out):
        """This method sets the gradients of the params in each layer
        of the network

        Args:
            d_out: The gradient of loss w.r.t output
        """
        
        d_out_value = d_out
        for layer in reversed(self.value_layers):
            d_out_value = layer.backward(d_out_value, self.optimizer.l2_lambda * 2)

        return d_out_value

    def get_params(self):
        out_params = []
        for layer in self.network:
            params = layer.get_params()
            if params is None:
                continue
                
            weights, biases = params
            out_params = out_params + weights.flatten().tolist()
            out_params = out_params + biases.flatten().tolist()
            

        return np.array(out_params)

    def train(self, training_examples):
        i = 0
        num = len(training_examples)
        loss_sum = 0
        for state, improved_policy, value in training_examples:
            if i % 50 == 0:
                print(f"Training example {i}/{num}")
            improved_policy = np.array(improved_policy)
            improved_policy = improved_policy.reshape((343, 1))
            
            # Compute the predicted policy and value
            predicted_policy, predicted_value = self.get_policy(state), self.get_value(state).item()

            # Compute the total loss
            total_loss = self.optimizer.loss_function(predicted_value, value,\
                                                      predicted_policy, improved_policy, self.get_params())

            loss_sum += total_loss
            # Compute the gradient of the output w.r.t total loss
            d_out_policy = -1 * np.divide(improved_policy, predicted_policy) / 343
            d_out_value = self.optimizer.value_coeff * -1 * (value - predicted_value)

            # Backpropagate the output gradient through the network
            d_shared = self.backward_policy(d_out_policy)
            d_shared += self.backward_value(d_out_value)

            self.backward_shared(d_shared)
            
            # Update the network parameters using the SGD optimizer
            for layer in self.network:
                params = layer.get_params()
                if params is None:
                    continue

                weights, biases = params
                d_weights, d_biases = layer.get_grads()
   
                weights = self.optimizer.update_params(weights, d_weights) # this works fine
                biases = self.optimizer.update_params(biases, d_biases)

                layer.set_params(weights, biases)

            # Zero out the gradients for the next iteration
            for layer in self.network:
                layer.zero_grad()
            
            i += 1
        
        print(f"AVERAGE LOSS: {loss_sum / len(training_examples)}")
        return self

    def save_network(self):
        save_model(self)
        