"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-24 15:29:06
@Description: 
"""

import tensorflow as tf
# 这段代码有点过时，🚫
# hello_tensor_flow = tf.constant("Hello, TensorFlow")
# sess = tf.session()
# print(sess.run(hello_tensor_flow))

with tf.compat.v1.Session() as sess:
    hello_tensor_flow = tf.constant("Hello, TensorFlow")
    print(sess.run(hello_tensor_flow))


with tf.compat.v1.Session() as sess:
    a = tf.compat.v1.placeholder(tf.int32)
    b = tf.compat.v1.placeholder(tf.int32)
    # 这个操作仅仅被定义好但尚未执行
    c = a + b
    values = {a: 5, b: 3}
    print(sess.run([c], values))


with tf.compat.v1.Session() as sess:
    a = tf.compat.v1.placeholder(tf.int32, name='a')
    b = tf.compat.v1.placeholder(tf.int32, name='b')
    c = tf.add(a, b, name='add')
    values = {a: 5, b: 3}
    summary_writer = tf.compat.v1.summary.FileWriter("/tmp/1", sess.graph)
