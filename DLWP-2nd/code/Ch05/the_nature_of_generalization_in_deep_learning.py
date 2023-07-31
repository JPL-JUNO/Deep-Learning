"""
@Description: 深度学习泛化的本质
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-31 22:06:25
"""

from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60_000, 28 * 28)).astype('float32') / 255.

random_train_labels = train_labels[:]  # 会自动（深）复制
np.random.shuffle(random_train_labels)

from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels,
          epochs=100, batch_size=128, validation_split=.2)
# 只要模型具有足够得表现力，就可以训练模型模拟任何数据
# 这里将标签打乱，模型在训练集上仍然可以训练出良好得准确率，但是验证集的准确率差的很
# 验证集的准确率不会提高，因为并不是有规律的打乱，不会说是3和7标签互换，这也就导致验证的时候准确率就是靠猜（10%）
# 模型最终只会记住特定的输入，就是字典一样
