"""
@Description: 提高模型容量
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-07 23:18:54
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60_000, 28 * 28)).astype('float32') / 255.

model = keras.Sequential([
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history_small_model = model.fit(
    train_images, train_labels,
    epochs=20, batch_size=128, validation_split=.2
)

import matplotlib.pyplot as plt
val_loss = history_small_model.history['val_loss']
epochs = range(1, 21)
plt.plot(epochs, val_loss, 'b--', label='Validation Loss')
# 模型的表现能力有限
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 这张图的验证没有出现峰值然后进行拐点，这表明可以拟合模型，但是没有办法实现过拟合
# 请记住，在任何情况下都可以实现过拟合
