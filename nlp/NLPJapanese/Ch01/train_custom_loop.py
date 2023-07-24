"""
@Description: 网络学习用的代码
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 16:42:46
"""

import sys
sys.path.append('..')
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2,
                    hidden_size=hidden_size,
                    output_size=3)
optimizer = SGD(lr=learning_rate)
data_size = len(x)
#  如果不能整除的话会丢掉一些数据
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []
for epoch in range(max_epoch):
    # 需要随机选择数据作为 mini-batch
    # 这里，我们以 epoch 为单位打乱数据，
    # 对于打乱后的数据，按顺序从头开始抽取数据。
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]
    for iters in range(max_iters):
        batch_x = x[iters * batch_size: (iters + 1) * batch_size]
        batch_t = t[iters * batch_size: (iters + 1) * batch_size]
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(
                f'|epoch {(epoch+1):03d} | iter {iters+1} / {max_iters} | loss {avg_loss:.3f}')
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(len(loss_list)), loss_list)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()

step = .001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')
# 因为数据被 permutation 了
x, t = spiral.load_data()
CLS_NUM = 3
N = 100
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i * N:(i + 1) * N, 0],
                x[i * N:(i + 1) * N, 1], s=40, marker=markers[i])
plt.show()
