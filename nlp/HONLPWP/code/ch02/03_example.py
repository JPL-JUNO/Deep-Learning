"""
@Title: 训练词袋分类器
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-23 22:32:32
@Description: 词袋作为文本的向量表示，其中每个向量维都能捕获文本中单词的出现频率、存在与否或加权值，但是不能捕获单词之间的顺序。
"""

import random
import nltk
from nltk.corpus import twitter_samples

pos_tweets = [(string, 1)
              for string in twitter_samples.strings("positive_tweets.json")]
neg_tweets = [(string, 0)
              for string in twitter_samples.strings("negative_tweets.json")]
# 正负样本连接起来
pos_tweets.extend(neg_tweets)
comb_tweets = pos_tweets
random.shuffle(comb_tweets)
tweets, labels = (zip(*comb_tweets))

from sklearn.feature_extraction.text import CountVectorizer
# 利用 sklearn 中的 CountVectorizer 函数生成特征并将特征数量限制为 10 000。我们
# 还使用了一元（unigram）和二元（bigram）特征。一个 n 元表示从文本中连续采样的 n 个单词特
# 征。一元模型是通常的单个单词特征，而二元模型则是文本中两个连续的单词序列。因为二元模
# 型是两个连续的单词，所以可以捕获文本中的短单词序列或者短语。
count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10_000)
X = count_vectorizer.fit_transform(tweets)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=10)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))

# 我们将使用 tfidf 向量化器对模型展开测试，该向量化
# 器似于基于计数的 n 元语法模型，不同之处在于它对计数进行加权：根据单词在所有文档或文本
# 中的出现情况为单词赋予权重。这意味着相较于只在特定文档中出现的单词，在所有文档里都频
# 繁出现的单词的权重会更低。
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)
X = tfidf.fit_transform(tweets)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=.2, random_state=0)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
# TfidfVectorizer 并没有提高模型的准确率 ❌

tfidf = TfidfVectorizer(ngram_range=(
    1, 2), max_features=10_000, stop_words=stopwords.words("english"))
X = tfidf.fit_transform(tweets)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=.2, random_state=0)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
# 对测试数据的评估显示：模型的准确率有所下降。去除停用词可能并不总会提高准确率，因
# 为准确性还取决于训练数据。特定的停用词可能会出现在指示推文情感的常见短语中。❌
