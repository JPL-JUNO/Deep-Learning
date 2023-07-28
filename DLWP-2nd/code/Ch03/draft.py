"""
@Description: 一些代码的草稿
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-28 19:52:55
"""

from tensorflow import keras
import numpy as np
num_samples_per_class = 1_000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, .5], [.5, 1]], size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, .5], [.5, 1]], size=num_samples_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype='float32'),
                     np.ones((num_samples_per_class, 1), dtype='float32')))

model = keras.Sequential([keras.layers.Dense(1)])
# 这些字符串是访问 python 对象的快捷方式
# model.compile(optimizer='rmsprop',
#               loss='mean_squared_error',
#               metrics=['accuracy'])
# fit 的效果好像很差
# 也可以把这些参数指定为实例对象
# This is useful if you want to pass your own custom losses or metrics,
# or if you want to further configure the objects you’re using—for instance,
# by passing a learning_rate argument to the optimizer:

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
#               loss=my_custom_loss,
#               metrics=[my_custom_metric_1, my_custom_metric_2])

# history = model.fit(inputs,
#                     targets,
#                     epochs=5,
#                     batch_size=128)
# print(history.history)

# Using the validation_data argument
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.1),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]
num_validation_samples = int(.3 * len(inputs))

val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]
model.fit(training_inputs,
          training_targets,  # 训练数据，用于更新模型权重
          epochs=5,
          batch_size=16,
          validation_data=(val_inputs, val_targets))  # 验证数据，仅用来监控验证损失和指标

# 这不是 fit 的输出最后一步嘛
loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)


# Inference: Using a model after training
# 一种简答的方法就是调用该模型（__call__）
# predictions = model(val_inputs)
# 但是这种方法一次性处理全部的输入，如果其中包含大量数据，那么这种方法是不可行的
# 一种更好的方法是调用 predict() 方法
predictions = model.predict(val_inputs, batch_size=128)
# 很奇怪，运行的结果超出了 [0, 1] 的范围
print(predictions[:10])
