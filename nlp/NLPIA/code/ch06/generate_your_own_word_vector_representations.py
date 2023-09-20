"""
@Title: 生成定制化词向量表示 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-26 13:36:42
@Description: 
"""

from gensim.models.word2vec import Word2Vec
# 表示词向量的元素个数
num_features = 300
# 模型中的最低词频，如果语料库较大，可以设置大一些，反之小一些
min_word_count = 1
# 训练使用的 CPU 核数，如果需要动态设置，使用：
# import multiprocessing
# num_workers = multiprocessing.cpu_count()
num_workers = 2
# 上下文的窗口大小
window_size = 6
# 高频词条的降采样率
subsampling = 1e-3

token_list = [
    ['to', 'provide', 'early', 'intervention/early', 'childhood', 'special',
     'education', 'services', 'to', 'eligible', 'children', 'and', 'their',
     'families'],
    ['essential', 'job', 'functions'],
    ['participate', 'as', 'a', 'transdisciplinary', 'team', 'member', 'to',
     'complete', 'educational', 'assessments', 'for']
]
model = Word2Vec(token_list,
                 workers=num_workers,
                 vector_size=num_features,
                 min_count=min_word_count,
                 window=window_size,
                 sample=subsampling)

# 一旦词向量模型训练完成，则可以通过冻结模型以及丢弃不必要的信息来减少大于一般的占用内存
# model.init_sims(replace=True)
# 但是在 Gensim 4.0.0 中，已经做了默认的优化，这将被移除
# The init_sims method will freeze the model, storing the weights of the hidden layer
# and discarding the output weights that predict word co-ocurrences. The output
# weights aren’t part of the vector used for most Word2vec applications. But the model
# cannot be trained further once the weights of the output layer have been discarded.

model_name = "my_domain_specific_word2vec_model"
model.save(model_name)

from gensim.models.word2vec import Word2Vec
model_name = "my_domain_specific_word2vec_model"
model = Word2Vec.load(model_name)
print(model.wv.most_similar("education", topn=2))
