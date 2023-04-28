"""
@Description: Tensors
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-04-28 22:21:38
"""

import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[7, 8, 9], [10, 11, 12]])

z = x + y
print(z)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
y = torch.tensor([[7, 8, 9], [10, 11, 12]], device=device)

z = x + y
print(z)
print(z.size())
print(z.device)

z = z.to("cpu")
print(z.device)

import numpy as np

w = torch.tensor([1, 2, 3])
w = torch.tensor((1, 2, 3))
w = torch.tensor(np.array([1, 2, 3]))
w = torch.empty(100, 200)  # uninitialized; element values are not predicted
w = torch.zeros(100, 200)
w = torch.ones(100, 200)

w = torch.rand(100, 200)  # a uniform distribution on the interval [0, 1)
# a normal distribution with a mean of 0 and a variance of 1
w = torch.randn(100, 200)
# random integers between 5 and 10
w = torch.randint(5, 10, (100, 200))
w = torch.empty((100, 200), dtype=torch.float64,
                device='cuda')
x = torch.empty_like(w)
