"""
@Title: 训练影评情感分类器
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-22 22:41:43
@Description: 
"""
from nltk.corpus import movie_reviews
import random
import nltk

# categories() 函数将返回 pos 或 neg，分别代表正面或负面情绪。
cats = movie_reviews.categories()
reviews = []
for cat in cats:
    for fid in movie_reviews.fileids(cat):
        review = (list(movie_reviews.words(fid)), cat)
        reviews.append(review)
random.shuffle(reviews)

all_word_in_reviews = nltk.FreqDist(wd.lower() for wd in movie_reviews.words())
top_word_in_reviews = [list(wds) for wds in zip(
    *all_word_in_reviews.most_common(2_000))][0]


def ext_ft(review, top_words):
    # 每条电影评论都将被传递给 ext_ft()函数并由该函数返回包含二进制特征的字典
    review_wds = set(review)
    ft = {}
    for wd in top_words:
        ft["word_present({})".format(wd)] = (wd in review_wds)
    return ft


feature_sets = [(ext_ft(d, top_word_in_reviews), c) for (d, c) in reviews]
train_set, test_set = feature_sets[200:], feature_sets[:200]

# 🍀
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(10)

# 下面使用 sklearn 中的随机森林来进行评估
from sklearn.feature_extraction import DictVectorizer
d_vect = None


def get_train_test(tr_set, te_set):
    """对训练数据转为 0-1 的编码格式"""
    global d_vect
    d_vect = DictVectorizer(sparse=True)
    X_tr, y_tr = zip(*tr_set)
    X_tr = d_vect.fit_transform(X_tr)
    X_te, y_te = zip(*te_set)
    X_te = d_vect.transform(X_te)
    return X_tr, y_tr, X_te, y_te


X_train, y_train, X_test, y_test = get_train_test(train_set, test_set)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,
                            n_jobs=-1, random_state=10)
rf.fit(X_train, y_train)
# 微小提升
from sklearn.metrics import accuracy_score
preds = rf.predict(X_test)
print(accuracy_score(preds, y_test))

# 我们利用 NLTK 停用词语料库来删除停用词，并像之前一样选择前 2000 个最为常见的单词
from nltk.corpus import stopwords
stopwords_list = stopwords.words("english")
all_word_in_reviews = nltk.FreqDist(
    word.lower() for word in movie_reviews.words() if word not in stopwords_list)
top_word_in_reviews = [list(words) for words in zip(
    *all_word_in_reviews.most_common(2_000))][0]
feature_sets = [(ext_ft(d, top_word_in_reviews), c) for d, c in reviews]
train_set, test_set = feature_sets[200:], feature_sets[:200]
X_train, y_train, X_test, y_test = get_train_test(train_set, test_set)
rf = RandomForestClassifier(n_estimators=100, random_state=10, n_jobs=-1)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
# 实际上，删除该数据集的停用词并没有提高模型的表现，反而使准确率有所降低。
print(accuracy_score(preds, y_test))

feature_list = zip(d_vect.get_feature_names_out(), rf.feature_importances_)
feature_list = sorted(feature_list, key=lambda x: x[1], reverse=True)
print(feature_list[:20])
# 虽然二进制特征可能对基本文本分类任务很有用，但是不适用于更为复杂的文本分类应用。
# 主要在于维度太大，权重的计算不需要涉及这么多的计算
