import numpy as np
from numpy import ndarray
from typing import Callable, List


def square(x: ndarray) -> ndarray:
    """
    将输入ndarray中的每个元素进行平方运算
    """
    return np.power(x, 2)


def leaky_relu(x: ndarray) -> ndarray:
    """
    将Leaky ReLU函数应用于ndarray中的每个元素
    """
    return np.maximum(.2 * x, x)


def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = .001) -> ndarray:
    '''
    计算函数func在input_数组中每个元素处的导数
    '''
    # 这个计算单点似乎可以，但是计算多元的偏导数有问题
    # 20230119，没有问题，对于数组的话，计算的数组元素对应的导数，并不是说多元函数的偏导数
    return (func(input_ + delta)-func(input_ - delta)) / (2 * delta)


# Callable[[ndarray], ndarray] in Python is a type hint that describes a function
# that takes a NumPy array as its input and returns another NumPy array as its output.
Array_function = Callable[[ndarray], ndarray]


# List[Array_function] is a type hint in Python that describes a list of functions
# that take a NumPy array as input and return a NumPy array as output.
Chain = List[Array_function]


def chain_length_2(chain: Chain, x: ndarray) -> ndarray:
    '''
    '''
    assert len(chain) == 2, "Length of input 'chain' should be 2"
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(x))


def sigmoid(x: ndarray) -> ndarray:
    '''
    将sigmoid函数应用于输入ndarray中的每个元素。
    '''
    return 1 / (1 + np.exp(-x))


def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    '''
    '''
    assert len(chain) == 2, \
        "This function requires 'Chain' objects of length 2"
    assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"
    f1 = chain[0]
    f2 = chain[1]

    f1_of_x = f1(input_range)
    df1dx = deriv(f1, input_range)

    df2du = deriv(f2, f1_of_x)

    return df1dx * df2du


def chain_length_3(chain: Chain,
                   x: ndarray) -> ndarray:
    '''
    '''
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(x)))


def chain_deriv_3(chain: Chain,
                  input_range: ndarray) -> ndarray:
    '''
    '''
    assert len(chain) == 3,\
        "This function requires 'Chain' objects to have length 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    f1_of_x = f1(input_range)

    f2_of_x = f2(f1_of_x)

    df3du = deriv(f3, f2_of_x)

    df2fu = deriv(f2, f1_of_x)

    df1du = deriv(f1, input_range)

    return df3du * df2fu * df1du


def plot_chain(ax,
               chain: Chain,
               input_range: ndarray,
               length: int = 2) -> None:
    '''
    '''
    assert input_range.ndim == 1,\
        'Function requires a 1 dimensional ndarray as input_range'
    # 计算复合函数的结果
    if length == 2:
        output_range = chain_length_2(chain, input_range)
    if length == 3:
        output_range = chain_length_3(chain, input_range)
    ax.plot(input_range, output_range)


def plot_chain_deriv(ax,
                     chain: Chain,
                     input_range: ndarray,
                     length: int = 2) -> ndarray:
    '''
    '''
    if length == 2:
        output_range = chain_deriv_2(chain, input_range)
    if length == 3:
        output_range = chain_deriv_3(chain, input_range)
    ax.plot(input_range, output_range)


def multiple_inputs_array(x: ndarray,
                          y: ndarray,
                          sigma: Array_function) -> ndarray:
    '''
    '''
    assert x.shape == y.shape

    a = x + y
    return sigma(a)


def multiple_inputs_add_backward(x: ndarray, y: ndarray, sigma: Array_function) -> float:
    '''
    '''
    a = x + y

    dsda = deriv(sigma, a)

    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady


def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
    '''
    '''
    assert X.shape[1] == W.shape[0], \
        '''
        对于矩阵乘法，第一个数组中的列数应与第二个数组中的行数相匹配。而参数中，
        第一个数组中的列数为{0}，第二个数组中的行数为{1}。
        '''.format(X.shape[1], W.shape[0])

    N = np.dot(X, W)
    return N


def matmul_backward_first(X: ndarray, W: ndarray) -> ndarray:
    '''
    Computes the backward pass of a matrix multiplication with respect to the first argument.
    '''
    dNdX = np.transpose(W, (1, 0))

    return dNdX


def matrix_forward_extra(X: ndarray, W: ndarray,
                         sigma: Array_function) -> ndarray:
    '''
    计算涉及矩阵乘法的函数（一个额外的函数）的前向传递结果。
    '''

    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)

    return S


def matrix_function_backward_1(X: ndarray, W: ndarray,
                               sigma: Array_function) -> ndarray:
    '''
    计算矩阵函数相对于第一个元素的导数
    '''

    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)

    S = sigma(N)

    dSdN = deriv(sigma, N)  # 1x1
    dNdX = np.transpose(W, (1, 0))

    return np.dot(dSdN, dNdX)


def matrix_function_forward_sum(X: ndarray, W: ndarray, sigma: Array_function) -> float:
    '''
    计算函数sigma的前向传递结果
    '''

    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)
    return L


def matrix_function_backward_sum_1(X: ndarray, W: ndarray, sigma: Array_function) -> ndarray:
    '''
    计算矩阵函数关于第一个矩阵输入的和的导数。
    '''
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)

    dLdS = np.ones_like(S)
    dSdN = deriv(sigma, N)

    dLdN = dLdS * dSdN

    dNdX = np.transpose(W, (1, 0))

    dLdX = np.dot(dLdN, dNdX)
    return dLdX
