import numpy as np


class StochasticGradientDescent:

    def __init__(self, l2_lambda=0.00001, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.value_coeff = 1
        # self.c=1e-3
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

        value_loss = 0.5 * ((true_value - pred_value) ** 2)
        policy_loss = -np.matmul(true_search_policy.T, np.log(pred_policy)).item()


        self.reg_term = self.l2_lambda * np.sum(params**2)
        loss = (self.value_coeff * value_loss) + policy_loss + self.reg_term
        # print(f"VALUE LOSS: {self.value_coeff * value_loss}")
        # print(f"POLICY LOSS: {policy_loss}")
        # print(f"REG LOSS: {self.reg_term}")

        return loss
    

class Adam:
    def __init__(self) -> None:
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.timestep = 0
        self.l2_lambda = 0.0001
        self.reg_term = 0

        self.m = None   # first moment
        self.v = None   # second moment

        # bias-corrected moment estimates
        self.m_hat = None      
        self.v_hat = None      


    def update_params(self, params, grads):
        """
        Parameters:
            params Parameters to be updated.
            grads: Gradients for each parameter.

        Returns:
            updated_parameters: The pdated parameters.
        """

        self.timestep += 1
    
        self.m = np.zeros_like(params)

        self.v = np.zeros_like(params)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        self.m_hat = np.zeros_like(params)

        self.m_hat = np.zeros_like(params)
        
        m_hat = self.m / (1 - self.beta1 ** self.timestep)
        v_hat = self.v / (1 - self.beta2 ** self.timestep)

        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

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

        value_loss = 0.5 * ((true_value - pred_value) ** 2)
        policy_loss = -np.matmul(true_search_policy.T, np.log(pred_policy)).item()

        self.reg_term = self.l2_lambda * np.sum(params**2)
        loss = value_loss + policy_loss + self.reg_term

        return loss

    

