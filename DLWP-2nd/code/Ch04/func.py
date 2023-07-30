"""
@Description: 提供一些函数支持
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-30 13:07:19
"""
import numpy as np
from numpy import ndarray


def vectorize_sequence(sequences, dimension: int = 10_000) -> ndarray:
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1
    return results


def to_one_hot(labels: ndarray | list, dimension: int = 50) -> ndarray:
    """实现独热编码

    Parameters
    ----------
    labels : _type_
        _description_
    dimension : int, optional
        维度变量, by default 50

    Returns
    -------
    _type_
        _description_
    """
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


if __name__ == '__main__':
    pass
