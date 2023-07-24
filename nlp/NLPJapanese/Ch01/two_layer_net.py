"""
@Description: 实现一个简单的神经网络
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 15:56:48
"""

import sys
sys.path.append('..')
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = .01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = .01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 因为 Softmax with Loss 层和其他层的处理方式不同，所以不将
        # 它放入 layers 列表中，而是单独存储在实例变量 loss_layer 中。
        self.loss_layer = SoftmaxWithLoss()
        self.params, self.grads = [], []
        for layer in self.layers:
            # layer.params 是一个 list
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        # 推理
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
