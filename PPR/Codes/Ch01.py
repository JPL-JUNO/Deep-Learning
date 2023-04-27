"""
@Description: An introduction to PyTorch
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-04-27 21:52:54
"""

import urllib.request
url = 'https://pytorch.tips/coffee'
fpath = 'coffee.jpg'
urllib.request.urlretrieve(url, fpath)

import matplotlib.pyplot as plt
from PIL import Image
