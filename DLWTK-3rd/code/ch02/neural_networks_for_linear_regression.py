"""
@Description: Simple linear regression using TensorFlow Keras
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-11 16:56:58
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as K
from tensorflow.keras.layers import Dense

np.random.seed(0)
area = 2.5 * np.random.randn(100) + 25
price = 25 * area + 5 + np.random.randint(20, 50, size=len(area))
data = np.array([area, price])
data = pd.DataFrame(data=data.T, columns=['area', 'price'])

plt.scatter(data['area'], data['price'])
plt.show()

# The input to neural networks should be normalized; this is because input gets multiplied with
# weights, and if we have very large numbers, the result of multiplication will be large, and soon
# our metrics may cross infinity (the largest number your computer can handle):
# 规范化，特征和目标都做了规范化
data = (data - data.min()) / (data.max() - data.min())
model = K.Sequential([
    Dense(1, input_shape=[1,], activation=None)
])
print(model.summary())

# To train a model, we will need to define the loss function and optimizer. The loss function defines
# the quantity that our model tries to minimize, and the optimizer decides the minimization
# algorithm we are using. Additionally, we can also define metrics, which is the quantity we want
# to log as the model is trained.
model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(x=data['area'], y=data['price'],
          epochs=1000, batch_size=32, verbose=1,
          validation_split=.2)

y_pred = model.predict(data['area'])
plt.plot(data['area'], y_pred, color='red', label='Predicted Price')
plt.scatter(data['area'], data['price'], label='Training Data')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()
