"""
@Description: CBOW 模型的实现
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-27 15:21:47
"""

import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        # 隐藏层应该小于输入层的神经元个数
        assert vocab_size > hidden_size
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = .01 * np.random.randn(V, H).astype('f')
        W_out = .01 * np.random.randn(H, V).astype('f')

        # 生成层
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 将该神经网络中使用的权重参数和梯度
        # 分别保存在列表类型的成员变量 params 和 grads 中。
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.word_vecs = W_in

    def forward(self, contexts, target):
        # 我们假定参数 contexts 是一个三维 NumPy 数组，比如说(6,2,7)的形状，
        # 其中第 0 维的元素个数是 mini-batch 的数量，
        # 第 1 维的元素个数是上下文的窗口大小，
        # 第 2 维表示 one-hot 向量。
        # 此外，target 是 (6,7) 这样的二维形状。
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * .5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= .5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
