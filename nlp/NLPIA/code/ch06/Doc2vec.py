"""
@Title: 利用 Word2Vec 计算文档的相似度
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-26 23:07:39
@Description: [Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053v2.pdf)
"""

import multiprocessing
import numpy as np
num_cores = multiprocessing.cpu_count()

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess

corpus = ["This is the first document",
          "another document"]

training_corpus = []
# 如果内存不足，并且在预先知道文档数量的情况下，可以为 training_corpus 使用预分配的 numpy 数组，而不是 Python 列表
training_corpus = np.empty(len(corpus), dtype=object)

for i, text in enumerate(corpus):
    tagged_doc = TaggedDocument(simple_preprocess(text), [i])
    training_corpus.append(tagged_doc)
model = Doc2Vec(size=100, min_count=2,
                workers=num_cores, iter=10)
model.build_vocab(training_corpus)
model.train(training_corpus, total_examples=model.corpus_count,
            epochs=model.iter)

# Doc2vec requires a “training” step when inferring new vectors.
model.infer_vector(simple_preprocess(
    "this is a completely unseen document"), steps=10)
