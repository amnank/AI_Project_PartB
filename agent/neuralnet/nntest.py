from nn import Layer, tanH, Sigmoid # pylint: disable=import-error
from cnn import Convolutional, Flatten # pylint: disable=import-error
import numpy as np

network = [
    Convolutional((14,7,7), 3, 20),
    tanH(),
    Convolutional((20,5,5), 3, 20),
    tanH(),
    Flatten((20,3,3), (180,1)),
    Layer(180, 100),
    Sigmoid()
]

policy_layers = [
    Layer(100, 343),
    Sigmoid()
]

value_layers = [
    Layer(100, 1),
    tanH()
]

inp = np.random.randn(14, 7, 7)

output = inp
for layer in network:
    output = layer.forward(output)
    print(output.shape)

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
