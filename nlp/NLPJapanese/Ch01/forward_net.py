"""
@Description: TwoLayerNet 代码
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 09:59:55
"""

import numpy as np
from numpy import ndarray


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x: ndarray):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W: ndarray, b: ndarray):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size: int,
                 hidden_size: int, output_size: int):
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layers = [Affine(W1, b1),
                       Sigmoid(),
                       Affine(W2, b2),
                       ]
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        def predict(self, x):
            for layer in self.layers:
                x = layer.forward(x)
            return x
