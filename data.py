import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

# suppress tensorflow logs (only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import mnist

# load the MNIST dataset
def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (training_intput, training_label), (test_input, test_label) = mnist.load_data()

    # transform to [0..1] values and 784-dimensional vectors
    training_intput  = np.reshape(training_intput / 255.0, (len(training_intput), 784, 1))
    test_input  = np.reshape(test_input / 255.0, (len(test_input), 784, 1))

    # create expected outputs: 10-dimensional vectors, 1 for the correct label, 0 otherwise
    training_expected = np.zeros((len(training_label), 10, 1))
    for i in range(len(training_label)):
        training_expected[i][training_label[i]][0] = 1

    test_expected = np.zeros((len(test_label), 10, 1))
    for i in range(len(test_label)):
        test_expected[i][test_label[i]][0] = 1
    
    return (training_intput, training_expected, test_input, test_expected)