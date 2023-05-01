from nn import *
from cnn import *
import numpy as np

network = [
    Convolutional((1,10,10), 3, 3),
    tanH(),
    Convolutional((3,8,8), 3, 2),
    tanH(),
    Flatten((2,6,6), (72,1)),
    Layer(72, 100),
    ReLU()
]

policy_layers = [
    Layer(100, 343),
    Sigmoid()
]

value_layers = [
    Layer(100, 1),
    tanH()
]

inp = np.random.randn(1, 10, 10)

output = inp
for layer in network:
    output = layer.forward(output)

print("POLICY: ")
policy = output
for layer in policy_layers:
    policy = layer.forward(policy)
print(policy)

print("VALUE: ")
value = output
for layer in value_layers:
    value = layer.forward(value)
print(value)
