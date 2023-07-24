"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 10:11:33
"""
import numpy as np
from numpy import ndarray


def softmax(x: ndarray) -> ndarray:
    if x.ndim == 2:
        # 减去一个最大值不会影响 softmax
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, t.size)
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # ✨
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
