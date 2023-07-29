"""
@Description: Our First example of TensorFlow code
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-29 21:44:26
"""

import tensorflow as tf
from tensorflow import keras
NB_CLASSES = 10
RESHAPED = 784

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPED,),
                             kernel_initializer='zeros',
                             name='dense_layer',
                             activation='softmax'))
