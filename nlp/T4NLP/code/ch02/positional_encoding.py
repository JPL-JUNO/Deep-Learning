"""
@Title: 位置编码
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-25 16:46:31
@Description: pe: position encoding
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
from sklearn.metrics.pairwise import cosine_similarity
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
# 词向量
a = model2.wv[word1]
b = model2.wv[word2]

if d_print == 1:
    print(a)
dot = np.dot(a, b)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
cos = dot / (norm_a * norm_b)

aa = a.reshape(-1, 512)
ba = b.reshape(-1, 512)
cos_lib = cosine_similarity(aa, ba)

# 存储位置编码
pe1 = aa.copy()
pe2 = aa.copy()

paa = aa.copy()
pba = ba.copy()
d_model = 512
# 每个词向量的维度
max_print = d_model
# 这个似乎是语料库的长度
max_length = 20

for i in range(0, max_print, 2):
    pe1[0][i] = math.sin(pos1 / (10_000**((2 * i) / d_model)))
    paa[0][i] = (paa[0][i] * math.sqrt(d_model) + pe1[0][i])

    pe1[0][i + 1] = math.cos(pos1 / (10_000**((2 * i) / d_model)))
    paa[0][i + 1] = (paa[0][i + 1] * math.sqrt(d_model) + pe1[0][i + 1])

    if d_print == 1:
        print(i, pe1[0][i], i + 1, pe1[0][i + 1])
        print(i, paa[0][i], i + 1, pe1[0][i + 1])
        print('\n')

# 一个使用 torch 的版本
max_len = max_length
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float()
                     * (-math.log(10_000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

for i in range(0, max_print, 2):
    pe2[0][i] = math.sin(pos2 / (10_000**((2 * i) / d_model)))
    pba[0][i] = (pba[0][i] * math.sqrt(d_model)) + pe2[0][i]

    pe2[0][i + 1] = math.cos(pos2 / (10_000**((2 * i) / d_model)))
    pba[0][i + 1] = (pba[0][i + 1] * math.sqrt(d_model)) + pe2[0][i + 1]

    if d_print == 1:
        print(i, pe2[0][i], i + 1, pe2[0][i + 1])
        print(i, paa[0][i], i + 1, paa[0][i + 1])
        print('\n')

print(word1, word2)
cos_lib = cosine_similarity(aa, ba)
print(cos_lib, "word similarity")
cos_lib = cosine_similarity(pe1, pe2)
print(cos_lib, "positional similarity")
cos_lib = cosine_similarity(paa, pba)
print(cos_lib, "positional encoding similarity")

if d_print == 1:
    print(word1)
    print("embedding")
    print(aa)
    print("positional encoding")
    print(pe1)
    print("encoded embedding")
    print(paa)

    print(word2)
    print("embedding")
    print(ba)
    print("positional encoding")
    print(pe2)
    print("encoded embedding")
    print(pba)
