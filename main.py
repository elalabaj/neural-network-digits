from network import Network
from dataset import load_dataset
import numpy as np

train_x, train_y, test_x, test_y = load_dataset()

network = Network([28*28, 128, 10])
network.train(test_x, test_y, train_x, train_y)
network.save()