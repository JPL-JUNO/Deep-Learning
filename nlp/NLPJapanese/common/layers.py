"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 10:09:27
"""

import numpy as np
from numpy import ndarray
from common.functions import softmax, cross_entropy_error


class SoftWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid 的输出
        self.t = None  # 监督标签

    def forward(self, x: ndarray, t: ndarray) -> float:
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        return self.loss

    # TODO 反向传播待写
    def backward(self, dout: float = 1):
        pass


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout: float):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # self.grads[0] = dW  # 浅复制
        self.grads[0][...] = dW  # 深复制
        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        # 这里将正向传播的输出保存在实例变量 out 中。
        # 然后，在反向传播中，使用这个 out 变量进行计算。
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, _ = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sum_dx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sum_dx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax 的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        # 在监督标签为 one-hot 向量的情况下，转换为正确标签的索引
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        # 这里突然理解稀疏矩阵如何计算的了，只需要明确哪一行取数就行
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        # 在反向传播时，从上一层（输出侧的层）传过来的梯度将原样传给下一层（输
        # 入侧的层）。不过，从上一层传来的梯度会被应用到权重梯度 dW 的特定行（idx），
        dW, = self.grads
        # 我们只需要更新权重 W，所有没有必要特意创建 dW（大小和 W 相同），相反
        # 只需要将其对应的梯度 dout 保存下来，就可以更新权重 W 的特定行，
        # 但是为了兼容已经实现的优化器类（optimizer）
        dW[...] = 0
        # 这种方式不太好
        # 因为会由多个反向传播加入一行，应该使用加入，而不是写入
        # dW[self.idx] = dout
        # for i, word_id in enumerate(self.idx):
        #     dW[word_id] += dout[i]
        # 或者这样写：
        np.add.at(dW, self.idx, dout)
        return None


class SigmoidWithLoss:
    pass
