"""
@Description: Anatomy of a neural network: Understanding core Keras APIs
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-27 22:35:17
"""
import tensorflow as tf
from tensorflow import keras


class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        # 创建权重
        input_dim = input_shape[-1]
        # add_weight() 是创建权重的快捷方法，也可以创建独立变量，并指定其作为层属性
        # self.W = tf.Variable(initial_value=tf.random.normal(
        #     shape=(input_dim, self.units)))
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros')

    def call(self, inputs):
        # 定义前向传播计算
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

    # __call__ 的大致代码如下：
    # def __call__(self, inputs):
    #     if not self.built:
    #         self.build(inputs.shape)
    #         self.built = True
    #     return self.call(inputs)


if __name__ == "__main__":
    my_dense = SimpleDense(units=32, activation=tf.nn.relu)
    input_tensor = tf.ones(shape=(2, 784))
    # 调用了 __call__ 方法
    output_tensor = my_dense(input_tensor)
    print(output_tensor.shape)  # (2, 32)
