"""
@Description: 各种节点的计算图
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 11:02:39
"""

import numpy as np
# repeat 节点的反向传播
D, N = 8, 7
# 输入
x = np.random.randn(1, D)
y = np.repeat(x, N, axis=0)  # 正向传播
dy = np.random.randn(N, D)  # 假设的梯度
dx = np.sum(dy, axis=0, keepdims=True)  # 反向传播

# sum 节点的反向传播
x = np.random.randn(N, D)
y = np.sum(x, axis=0, keepdims=True)
dy = np.random.randn(1, D)
dx = np.repeat(dy, N, axis=0)
