import numpy as np
from agent_network import AgentNetwork


class StochasticGradientDescent:
    """
    Updates the parameters of the nn according to SGD.
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = None
    
    def update_params(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        
        for i in range(len(params)):
            self.velocities[i] = self.momentum * self.velocities[i] + (1 - self.momentum) * grads[i]
            params[i] -= self.learning_rate * self.velocities[i]