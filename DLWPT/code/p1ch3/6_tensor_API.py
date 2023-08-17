"""
@Description: 张量 API
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-17 17:25:36
"""

import torch
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
assert a.shape == (3, 2)
assert a_t.shape == (2, 3)

# transpose()函数也可以作为张量的一个方法
a_t = a.transpose(0, 1)
