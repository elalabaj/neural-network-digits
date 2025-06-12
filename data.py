import os
import matplotlib.pyplot as plt

# suppress tensorflow logs (only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# load the MNIST dataset
# x - input, y - expected output
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# display a couple of first images
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
    
plt.tight_layout()
plt.show()