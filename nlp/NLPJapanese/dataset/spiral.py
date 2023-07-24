"""
@Description: 提供了几个便于处理数据集的类
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 15:18:16
"""

import numpy as np


def load_data(seed: int = 1984):
    np.random.seed(seed)
    N = 100  # 各类的样本数
    DIM = 2  # 数据的元素个数
    CLS_NUM = 3  # 类别数

    x = np.zeros(shape=(N * CLS_NUM, DIM))
    t = np.zeros(shape=(N * CLS_NUM, CLS_NUM), dtype=np.int)  # 监督标签

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * .2
            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix, j] = 1
    return x, t


if __name__ == '__main__':
    pass
