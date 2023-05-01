from nn import *
from cnn import *
import numpy as np

network = [
    Layer(2, 3),
    ReLU(),
    Layer(3, 100),
    ReLU(),
    Layer(100, 3),
    ReLU(),
    Layer(3, 1),
    tanH()
]

inp = np.array([0.5, -7]).reshape(-1, 1)

output = inp
for layer in network:
    output = layer.forward(output)
    
print(output)
