import os
import numpy as np

def save_model(model):
    if not os.path.exists(model.network_name):
        os.makedirs(model.network_name)

    for layer in model.network:
        if layer.get_params() is None:
            continue

        weights, biases = layer.get_params()

        w_fname = os.path.join(model.network_name, f"{layer.name}_weights.npy")
        b_fname = os.path.join(model.network_name, f"{layer.name}_biases.npy")

        np.save(w_fname, weights)
        np.save(b_fname, biases)


def load_model(model, load_network_name):
    for layer in model.network:
        if layer.get_params() is None:
            continue

        w_fname = os.path.join(load_network_name, f"{layer.name}_weights.npy")
        b_fname = os.path.join(load_network_name, f"{layer.name}_biases.npy")

        weights = np.load(w_fname)
        biases = np.load(b_fname)

        layer.set_params(weights, biases)


    