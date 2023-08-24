"""
@Title: 建立一个跳字模型
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-24 16:49:36
@Description: 跳字通过使用上下文中的单词来学习预测目标单词
"""

# TensorFlow 的 projector 模块为我们提供了在 TensorBoard 上添加词向量以进行可视化所
# 需的方法。
from tensorboard.plugins import projector
import os
import numpy as np
import tensorflow as tf

model_params = {
    "vocab_size": 50_000,  # 最大单词数
    "batch_size": 64,  # 各个训练步的批大小
    "embedding_size": 200,  # 词嵌入向量维度
    "num_negative": 64,  # 否定词采样数
    "learning_rate": 1.0,  # 训练学习率
    "num_train_steps": 500_000,  # 模型训练步数
}


class Word2vecModel:
    """为 Word2vec 模型初始化参数"""

    def __init__(self, data_set, vocab_size,
                 embed_size, batch_size, num_sampled, learning_rate,
                 ):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(
            initial_value=tf.constant(0),
            trainable=False,
            name="global_step",
        )
        self.skip_step = model_params["skip_step"]
        self.data_set = data_set


data_set = tf.data.Dataset.from_generator(generator,
                                          (tf.int32, tf.int32),
                                          (tf.TensorShape(
                                              model_params["batch_size"]),
                                           tf.TensorShape(model_params["batch_size"])),
                                          )
def generator:
    yield from batch_generator(model_params["vocab_size"],
                               model_params["batch_size"],
                               model_params["skip_window"],
                               file_params["visualization_folder"])