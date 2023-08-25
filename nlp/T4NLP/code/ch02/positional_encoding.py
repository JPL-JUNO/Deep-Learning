"""
@Title: 位置编码
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-25 16:46:31
@Description: 
"""

import math
# d_model = 512
# 伪代码
# def positional_encoding(pos, pe):
#     for i in range(0, 512, 2):
#         pe[0][i] = math.sin(pos / (10_000**((2 * i) / d_model)))
#         pe[0][i + 1] = math.sin(pos / (10_000**((2 * i) / d_model)))
#     return pe

import torch
import nltk
nltk.download("punkt")

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings(action="ignore")

d_print = 0
sample = open("text.txt", "r")
s = sample.read()
sample.close()

f = s.replace("\n", " ")

data = []
for i in sent_tokenize(f):
    # 通过一些标点符号进行分句
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)

# 创建 Skip-gram 模型
# sg : {0, 1}, optional
# Training algorithm: 1 for skip-gram; otherwise CBOW.
model2 = gensim.models.Word2Vec(data, min_count=1,
                                vector_size=512, window=5, sg=1)
word1 = "black"
word2 = "brown"
pos1 = 2
pos2 = 10
a = model2.wv[word1]
b = model2.wv[word2]

if d_print == 1:
    print(a)
dot = np.dot(a, b)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
cos = dot / (norm_a * norm_b)
