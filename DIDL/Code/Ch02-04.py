import numpy as np
from d2l import torch as d2l
from matplotlib_inline import backend_inline
from typing import function


def f(x: float) -> float:
    return 3 * x**x - 4 * x


def numerical_lim(f: function, x: float, h: float) -> float:
    """计算一个一元函数的在某个点处的导数

    Args:
        f (function): 需要计算导数
        x (float): _description_
        h (float): _description_

    Returns:
        float: _description_

    """
    return (f(x + h) - f(x - h)) / (2 * h)
