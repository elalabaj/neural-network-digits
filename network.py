from typing import List, Tuple
import numpy as np
import pickle

class Network:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(j, k) for k, j in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(j, 1) for j in layer_sizes[1:]]

    def save(self, filename = 'network.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)

    def load(self, filename = 'network.pkl'):
        with open(filename, 'rb') as f:
            self.weights, self.biases = pickle.load(f)

    # returns the activations and raw activations of the network
    def feedforward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        activations = [x.copy()]
        z_values = []
        for w, b in zip(self.weights, self.biases):
            z = w @ activations[-1] + b
            a = sigmoid(z)
            activations.append(a.copy())
            z_values.append(z.copy())
        return (activations, z_values)
    
    def train(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, num_epochs: int = 10, learning_rate: float = 0.01):
        for i in range(num_epochs):
            self.gradient_descent(train_x, train_y, learning_rate=learning_rate)
            print(f"Epoch #{i}: ", end='')
            self.test(test_x, test_y)

    def test(self, test_x: np.ndarray, test_y: np.ndarray):
        correct = 0
        for j in range(len(test_x)):
            output = self.feedforward(test_x[j])[0][-1]
            if np.argmax(output) == np.argmax(test_y[j]):
                correct += 1
        accuracy = correct / len(test_x)
        print(f"Accuracy: {accuracy:.4f}")

    # stochastic gradient descent algorithm, x - inputs array, y - expected outputs array
    def gradient_descent(self, x: np.ndarray, y: np.ndarray, minibatch_size: int = 10, learning_rate: float = 0.1):
        n = len(x)

        indices = np.arange(n)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        for i in range(n // minibatch_size):
            nabla_w = [np.zeros_like(w) for w in self.weights]
            nabla_b = [np.zeros_like(b) for b in self.biases]

            for j in range(i * minibatch_size, (i+1) * minibatch_size):
                nabla_w_prime, nabla_b_prime = self.backpropagation(x[j], y[j])

                for k in range(len(nabla_w)):
                    nabla_w[k] += nabla_w_prime[k]
                    nabla_b[k] += nabla_b_prime[k]
        
            for k in range(len(nabla_w)):
                self.weights[k] -= (learning_rate / minibatch_size) * nabla_w[k]
                self.biases[k] -= (learning_rate / minibatch_size) * nabla_b[k]
        
    # calculate pratial derivatives of the cost function
    # in respect to all the weights and biases
    def backpropagation(self, x: np.ndarray, y: np.ndarray):
        activations, z_values = self.feedforward(x)
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]
        
        error = self.cost_derivative(activations[-1], y) * sigmoid_derivative(z_values[-1])
                
        nabla_w[-1] =  error @ activations[-2].T
        nabla_b[-1] = error.copy()

        for i in range(2, self.num_layers):
            l = -i
            error = (self.weights[l+1].T @ error) * sigmoid_derivative(z_values[l])
            nabla_w[l] = error @ activations[l-1].T
            nabla_b[l] = error.copy()

        return (nabla_w, nabla_b)

    def cost(self, a_last: np.ndarray, y: np.ndarray) -> float:
        return 0.5 * sum((a_last - y) ** 2)
    
    def cost_derivative(self, a_last: np.ndarray, y: np.ndarray) -> np.ndarray:
        return a_last - y
    

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1.0 - sigmoid(x))