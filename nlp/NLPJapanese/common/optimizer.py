"""
@Description: 更新权重和偏置的优化器
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 15:03:18
"""


class SGD:
    def __init__(self, lr: float = .01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
