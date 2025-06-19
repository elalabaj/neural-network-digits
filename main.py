from network import Network
from dataset import load_dataset
import numpy as np
from draw import DigitRecognizer

train_x, train_y, test_x, test_y = load_dataset()

network = Network([28*28, 128, 64, 10])
network.load()
# network.train(test_x, test_y, train_x, train_y, 20, 0.1)
# network.save()

# network.test(test_x, test_y)

app = DigitRecognizer(network)
app.run()