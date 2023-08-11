"""
@Description: 简单的线性回归
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-11 16:18:55
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
area = 2.5 * np.random.randn(100) + 25
price = 25 * area + 5 + np.random.randint(20, 50, size=len(area))
data = np.array([area, price])
data = pd.DataFrame(data=data.T, columns=['area', 'price'])
# data = pd.DataFrame(data={'area': area, 'price': price})
plt.scatter(data['area'], data['price'])
plt.show()

W = sum(price * (area - np.mean(area))) / sum((area - np.mean(area))**2)
b = np.mean(price) - W * np.mean(area)
print('The regression coefficients are', W, b)

y_pred = W * area + b
plt.plot(area, y_pred, color='red', label='Predicted Area')
plt.scatter(data['area'], data['price'], label='Training Data')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()
