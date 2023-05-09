import numpy as np


class StochasticGradientDescent:

    def __init__(self, l2_lambda=0.001, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.c=1e-4
        self.reg_term = 0
    
    def update_params(self, params, grads):
        params += self.learning_rate * grads
        return params


    def loss_function(self, pred_value, true_value, pred_policy, true_search_policy, params):
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

        true_search_policy = np.array(true_search_policy)
        true_search_policy = true_search_policy.reshape((343, 1))

        value_loss = 0.5 * ((true_value - pred_value) ** 2)
        policy_loss = -np.matmul(true_search_policy.T, np.log(pred_policy)).item()

        self.reg_term = self.c * np.sum([np.sum(param**2) for param in params])
        loss = value_loss + policy_loss + self.reg_term

        return loss