import numpy as np
from agent_network import AgentNetwork # pylint: disable=import-error
from alpha_zero_logic import SelfPlay # pylint: disable=import-error

alpha_zero = SelfPlay()
network = alpha_zero.train_network(should_dump=True)

print("Finished training!")
