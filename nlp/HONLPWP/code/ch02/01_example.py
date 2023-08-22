"""
@Title: 训练词性标注器
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-22 19:31:50
@Description: 使用 NLTK 的标注集语料库和 sklearn 随机森林机器学习模型来训练自己的词性标注器
"""
# 1. 整个过程将是一个分类任务，因为我们需要预测句子中给定单词的词性标签
# 2. 使用带有词性标注的 NLTK treebank 数据集作为训练数据或标注数据，并提取单词的前缀、后缀以
# 3. 及文本中的前序单词和相邻单词作为训练的特征

import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def sentence_features(st: list, idx: int) -> dict:
    """计算一个一个单词在一个句子中的一些特征，包括前缀1， 前缀2，前缀3，后缀1，后缀2， 后缀3，前后一个单词，是否是大写开头，所在的索引与从后向前的索引。

    每个句子以 Python 列表的形式与当前单词的索引一起被传入，以提取该单词的特征，其中索引 idx 用于获取相邻单词的特征以及单词的前缀/后缀。

    Parameters
    ----------
    st : list
        词元组成的句子
    idx : int
        带判断的索引

    Returns
    -------
    dict
        特征字典
    """
    d_ft = {}
    d_ft["word"] = st[idx]
    d_ft["dist_from_first"] = idx - 0
    d_ft["dist_from_last"] = len(st) - idx
    # 判断是不是大写字母开头
    d_ft['capitalized'] = st[idx][0].upper() == st[idx][0]
    d_ft["prefix1"] = st[idx][0]
    d_ft["prefix2"] = st[idx][:2]
    d_ft["prefix3"] = st[idx][:3]
    d_ft["suffix1"] = st[idx][-1:]
    d_ft["suffix2"] = st[idx][-2:]
    d_ft["suffix3"] = st[idx][-3:]
    d_ft["prev_word"] = "" if idx == 0 else st[idx - 1]
    d_ft["next_word"] = "" if idx == (len(st) - 1) else st[idx + 1]
    d_ft["numeric"] = st[idx].isdigit()
    return d_ft


def get_untagged_sentence(tagged_sentence: list[tuple]) -> list[str]:
    # 🚩
    [s, _] = zip(*tagged_sentence)
    return list(s)


def ext_ft(tg_sent):
    sent, tag = [], []
    for tg in tg_sent:
        for index in range(len(tg)):
            sent.append(sentence_features(get_untagged_sentence(tg), index))
            tag.append(tg[index][1])
    return sent, tag


tagged_sentences = nltk.corpus.treebank.tagged_sents(tagset="universal")

X, y = ext_ft(tagged_sentences)

n_sample = 50_000
dict_vectorizer = DictVectorizer(sparse=True)
X_transformed = dict_vectorizer.fit_transform(X[:n_sample])
y_sampled = y[:n_sample]

# 这里有一点的问题，划分数据集之前进行了变换，实际上会高估模型的性能
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y_sampled, test_size=.2, random_state=123)

rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X_train, y_train)


def predict_pos_tags(sentence):
    features = [sentence_features(sentence, index)
                for index in range(len(sentence))]
    features = dict_vectorizer.transform(features)
    tags = rf.predict(features)
    return zip(sentence, tags)


test_sentence = "This is a sample POS tagger"
for tagged in predict_pos_tags(test_sentence.split()):
    print(tagged)

predictions = rf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(acc)

conf_matrix = confusion_matrix(y_test, predictions)
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10, 10))
plt.xticks(np.arange(len(rf.classes_)), rf.classes_)
plt.yticks(np.arange(len(rf.classes_)), rf.classes_)
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.show()

feature_list = zip(
    dict_vectorizer.get_feature_names_out(), rf.feature_importances_
)
sorted_features = sorted(feature_list, key=lambda x: x[1], reverse=True)
print(sorted_features[0:20])
# sorted_features[0:20]
