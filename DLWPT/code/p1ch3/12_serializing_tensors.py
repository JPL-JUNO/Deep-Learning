"""
@Description: 序列化张量
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-18 09:55:55
"""

import torch
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
torch.save(points, 'ourpoints.t')
# 作为替代方法，我们可以传递一个文件描述符来代替文件名：
with open('ourpoints.t', "wb") as f:
    torch.save(points, f)

# 加载张量 points 同样可以通过一行代码来实现：
points = torch.load("ourpoints.t")
with open('ourpoints.t', "rb") as f:
    points = torch.load(f)

# 如果我们只是想用 PyTorch 加载张量的话，我们可以用这种方法快速地保存张量，但是文件
# 格式本身是不具有互用性的，即我们无法用除 PyTorch 之外的软件读取张量。根据用例的不同，
# 这可能是一种局限，也可能不是，但我们应该学习如何在需要的时候以一种可互用的方式来保存
# 张量。

# 将张量 points 转换为一个 NumPy 数组（如前所述，这不会带来开销），同时将其
# 传递给 create_dataset()函数
import h5py
f = h5py.File("ourpoints.hdf5", 'w')
dset = f.create_dataset("coords", data=points.numpy())
f.close()

# 这里的“coords”是保存到 HDF5 文件的一个键，我们可以有其他键，甚至可以嵌套键。HDF5
# 中有趣的事情之一是，我们可以在磁盘上索引数据集，并只访问我们感兴趣的元素。假设我们只
# 想加载数据集中的最后 2 个点：
f = h5py.File("ourpoints.hdf5", "r")
dset = f["coords"]
last_points = dset[-2:]
# 当进行打开文件或需要数据集对象操作时，不会加载数据。更确切地说，在我们请求数据集
# 中第 2 行到最后一行数据之前，数据一直保存在磁盘上。此时，h5py 访问这两列并返回一个类
# 似 NumPy 数组的对象，该对象将所访问的区域封装在数据集中，其行为类似于 NumPy 数组，并
# 与其具有相同的 API。
# 因此我们要将返回的对象传递给 torch.from_numpy()函数直接获得张量。注意，在这种情况
# 下，数据会被复制到张量所在的存储中：
# 一旦完成数据加载，就关闭文件。关闭 HDF5 文件会使数据集失效，然后试图访问 dset 会抛出一个异常。
# 需要手动关闭文件，（不是说数据加载完就会自动关闭文件）

last_points = torch.from_numpy(dset[-2:])
f.close()
