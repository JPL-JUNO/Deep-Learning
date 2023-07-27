"""
@Description: CBOW 模型的推理
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-27 11:36:43
"""

import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul
# 样本的上下文数据
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 权重的初始化
W_in = np.random.randn(7, 3)  # 类似于编码
W_out = np.random.randn(3, 7)  # 类似于解码

# 生成层
# 生成与上下文单词数量等量（这里是两个）的处理输入层的 MatMul 层，输出侧仅
# 生成一个 MatMul 层。需要注意的是，输入侧的 MatMul 层共享权重 W_in。
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 正向传播
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
# h = np.vstack((h0, h1)).mean(axis=0)
h = .5 * (h0 + h1)
s = out_layer.forward(h)

print(s)
