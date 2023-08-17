"""
@Description: 从张量开始
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-17 17:17:42
"""
import torch

# 3.5.3 管理张量的 dtype 属性
# 可以指定适当的 dtype 作为构造函数的参数
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)

# 还可以使用相应的转换方法将张量创建函数的输出转换为正确的类型：
double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

# 或者使用更方便的方法：
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)
# 在底层，to()方法会检查转换是否是必要的，如果必要，则执行转换。
# 以 dtype 命名的类型转换方法（如 float()）是 to()的简写，但 to()方法可以接收其他参数


points_64 = torch.rand(5, dtype=torch.double)
points_short = points_64.to(torch.short)
print(points_64 * points_short)
