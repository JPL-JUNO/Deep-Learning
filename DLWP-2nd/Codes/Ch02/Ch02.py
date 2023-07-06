from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from numpy import ndarray

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

assert train_images.shape == (60000, 28, 28)
assert len(train_labels) == 60000
assert test_images.shape == (10000, 28, 28)
assert len(test_labels) == 10000


network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
assert train_images.ndim == 3
assert train_images.dtype == 'uint8'

digit = train_images[7]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)

my_slice = train_images[10:100]
assert my_slice.shape == (90, 28, 28)

batch = train_images[:128]
batch = train_images[128:256]


def naive_relu(x):
    assert x.ndim == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add(x, y):
    assert x.ndim == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


def naive_vector_dot(x: ndarray, y: ndarray) -> float:
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.shape == y.shape

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] + y[i]
    return z


def naive_matrix_vector_dot(x: ndarray, y: ndarray) -> ndarray:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
        # z[i] = naive_vector_dot(x[i, :], y)
    return z


def naive_matrix_dot(x, y):
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            row_x = x[i, :]
            col_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, col_y)
    return z
