"""
@Title: 卷积神经网络（CNN）
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-28 22:13:28
@Description: 
"""
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPool1D  # 卷积和池化层
import glob
import os
from random import shuffle


def pre_process_data(file_path):
    positive_path = os.path.join(file_path, 'pos')
    negative_path = os.path.join(file_path, 'neg')
    pos_label = 1
    neg_label = 0
    dataset = []
    for path, label in [(positive_path, pos_label), (negative_path, neg_label)]:
        for file_name in glob.glob(os.path.join(path, "*.txt")):
            with open(file_name, 'r', encoding="utf-8") as f:
                dataset.append((label, f.read()))
    shuffle(dataset)
    return dataset


dataset = pre_process_data('../../data/aclImdb/train')
