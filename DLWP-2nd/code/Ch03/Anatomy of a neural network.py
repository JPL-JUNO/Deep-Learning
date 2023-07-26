"""
@Description: Anatomy of a neural network
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-21 22:00:54
"""

from tensorflow import keras, Tensor
import tensorflow as tf


class SimpleDense(keras.layers.Layer):
    def __init__(self, units: int, activation=None) -> None:
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape: tuple) -> None:
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer='random_normal')
        self.b = self.add_weight(shape=(self.units),
                                 initializer='zeros')

    def call(self, inputs: Tensor) -> Tensor:
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


my_dense = SimpleDense(units=32, activation=tf.nn.relu)
input_tensor = tf.ones(shape=(2, 784))
output_tensor = my_dense(input_tensor)

from tensorflow.keras import layers
layer = layers.Dense(32, activation="relu")
