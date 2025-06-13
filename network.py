from typing import List
import numpy as np
import pickle

class Network:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(j, k) for k, j in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(j, 1) for j in layer_sizes[1:]]

    def save(self, filename = 'network.pkl'):
        # np.savez(filename, weights = self.weights, biases = self.biases, allow_pickle=True)
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)

    def load(self, filename = 'network.pkl'):
        # data = np.load(filename, allow_pickle=True)
        # self.weights = data['weights']
        # self.biases = data['biases']
        with open(filename, 'rb') as f:
            self.weights, self.biases = pickle.load(f)

    # returns the output of a network, when a are the activations in the first layer
    def feedforward(self, a: np.ndarray) -> np.ndarray:
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a
    

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))