"""
@Description: 将张量存储到 GPU
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-18 09:39:17
"""

import torch
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# 通过指定构造函数的相应参数在 GPU 上创建一个张量：
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
# 使用 to()方法将在 CPU 上创建的张量复制到 GPU 上
points_gpu = points.to(device="cuda")
# 如果我们的机器有多个 GPU，我们也可以通过从 0 开始传递一个整数来确定存储张量的 GPU
points_gpu = points.to(device="cuda:0")

points = 2 * points
points_gpu = 2 * points.to(device="cuda")
# 一旦计算出结果，张量 points_gpu 不会返回到 CPU

points_gpu = points_gpu + 4
# 为了将张量移回 CPU，我们需要向 to()方法提供一个 cpu 参数
points_cpu = points_gpu.to(device="cpu")
# 也可以使用简写的 cpu()和 cuda()方法来代替 to()方法实现相同的目标：
points_gpu = points.cuda()
points_gpu = points.cuda(0)  # GPU 索引默认为0
points_cpu = points_gpu.cpu()
# 还值得一提的是，通过使用 to() 方法，我们可以同时通过 device 和 dtype 参数来更改位置和数据类型。
# 使用 dtype 指定类型就是调用 to() 方法
