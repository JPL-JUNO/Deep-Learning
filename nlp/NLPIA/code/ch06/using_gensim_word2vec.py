"""
@Title: How to use the gensim.word2vec module
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-25 22:50:00
@Description: 'Word2Vec' object has no attribute 'most_similar'似乎没有这个属性了，需要添加 wv
"""

# from nlpia.data.loaders import get_data
# 可以用下面的命令来下载一个预训练的 Word2Vec 模型
# word_vectors = get_data("word2vec")

from gensim.models.keyedvectors import KeyedVectors
# 词向量会占用大量内存空间，如果可用内存有限，可以设置 limit 参数
# word_vectors = KeyedVectors.load_word2vec_format(
#     '../GoogleNews-vectors-negative300.bin.gz', binary=True)

word_vectors = KeyedVectors.load_word2vec_format(
    "../GoogleNews-vectors-negative300.bin.gz", limit=200_000, binary=True)

# The keyword argument positive takes a list of the vectors to be added together,
word_vectors.most_similar(positive=["cooking", "potatoes"], topn=5)
word_vectors.most_similar(positive=["germany", "france"], topn=1)

# Word vector models also allow you to determine unrelated terms.
# To determine the most unrelated term of the list, the method returns the term with
# the highest distance to all other list terms.
word_vectors.doesnt_match("potatoes milk cake computers".split())

# 可以完成计算 king + woman - man = queen
word_vectors.most_similar(positive=["king", "woman"], negative=["man"], topn=2)

# The gensim library also allows you to calculate the similarity between two terms.
word_vectors.similarity('princess', 'queen')

# If you want to develop your own functions and work with the raw word vectors, you
# can access them through Python’s square bracket syntax ([]) or the get() method
# on a KeyedVector instance.
word_vectors["phone"]
