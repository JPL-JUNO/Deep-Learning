"""
@Description: 改进模型拟合
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-07 23:00:22
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60_000, 28 * 28)).astype('float32') / 255.

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer=keras.optimizers.RMSprop(1.),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,
                    batch_size=128, validation_split=.2)

import matplotlib.pyplot as plt
plt.plot(range(1, 11), history.history['val_accuracy'])
plt.plot(range(1, 11), history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Acc.")
plt.title('Training with an incorrectly high learning rate')
plt.show()

model.compile(optimizer=keras.optimizers.RMSprop(1e-2),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,
                    batch_size=128, validation_split=.2)

import matplotlib.pyplot as plt
plt.plot(range(1, 11), history.history['val_accuracy'])
plt.plot(range(1, 11), history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Acc.")
plt.title('Training with an incorrectly high learning rate')
plt.show()
