"""
@Description: 一个端到端的案例：实现一个简单的分类器
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-26 23:27:35
"""

import tensorflow as tf
from tensorflow import Tensor
import numpy as np
from numpy import ndarray

num_per_class = 1_000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, .5], [.5, 1]], size=num_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, .5], [.5, 1]], size=num_per_class)

targets = np.vstack((np.zeros(shape=(num_per_class, 1)),
                     np.ones(shape=(num_per_class, 1))))
# 不指定 float32 的会报错
features = np.vstack((negative_samples, positive_samples)).astype(np.float32)
import matplotlib.pyplot as plt
# plt.scatter(features[:, 0], features[:, 1], c=targets)
# plt.show()
input_size = 2
output_size = 1
# 使用正态初始化似乎不是一个很好的开始（仅针对这个问题）
W = tf.Variable(initial_value=tf.random.uniform(
    shape=(input_size, output_size)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_size, )))


def model(features: ndarray) -> ndarray:
    # 模型给出预测值
    return tf.matmul(features, W) + b


def square_loss(targets: ndarray, predictions: ndarray) -> float:
    # 计算预测值和目标值之间的平均损失
    per_samples_loss = tf.square(targets - predictions)
    return tf.reduce_mean(per_samples_loss)


learning_rate = .1


def train(features, targets) -> float:
    # 训练函数（学习函数）用于更新参数
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss


losses = []
steps = 40
for step in range(steps):
    loss = train(features, targets)
    losses.append(loss)
    print(f'Loss at step {step:02d}: {loss:.4f}')
plt.plot(range(steps), losses, 'o')
plt.plot(range(steps), losses)
plt.show()
