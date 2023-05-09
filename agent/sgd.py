import numpy as np
from agent_network import AgentNetwork # pylint: disable=import-error


class StochasticGradientDescent:

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = None
    
    def update_params(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        
        for i,_ in enumerate(params):
            self.velocities[i] = self.momentum * self.velocities[i] + (1 - self.momentum) * grads[i]
            params[i] -= self.learning_rate * self.velocities[i]


    def loss_function(self, pred_value, true_value, pred_policy, true_search_policy, c=1e-4):
        """
        Computes the loss function for AlphaZero self-play reinforcement learning algorithm.

        Args:
        pred_value: predicted value of the current state by neural network
        true_value: final reward for the current state

        pred_policy: predicted probability distribution of the next move by neural network
        true_search_policy: search probabilities from MCTS

        Returns:
        loss: total loss
        """

        value_loss = (true_value - pred_value) ** 2
        policy_loss = -(true_search_policy * np.log(pred_policy))
        # reg_term = c * np.sum([np.sum(param**2) for param in parameters])
        loss = value_loss + policy_loss #+ reg_term

        return loss