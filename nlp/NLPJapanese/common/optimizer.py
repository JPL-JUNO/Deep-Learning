"""
@Description: 更新权重和偏置的优化器
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 15:03:18
"""
import numpy as np


class SGD:
    def __init__(self, lr: float = .01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Adam:
    def __init__(self, lr=0.001, beta_1=.9, beta_2=.999) -> None:
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta_2 **
                                 self.iter) / (1.0 - self.beta_1**self.iter)
        for i in range(len(params)):
            self.m[i] += (1 - self.beta_1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta_2) * (grads[i]**2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
