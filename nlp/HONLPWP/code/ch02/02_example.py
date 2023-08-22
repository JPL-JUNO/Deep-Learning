"""
@Title: 训练影评情感分类器
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-22 22:41:43
@Description: 
"""
from nltk.corpus import movie_reviews
import random

# categories() 函数将返回 pos 或 neg，分别代表正面或负面情绪。
cats = movie_reviews.categories()
reviews = []
for cat in cats:
    for fid in movie_reviews.fileids(cat):
        review = (list(movie_reviews.words(fid)), cat)
        reviews.append(review)
random.shuffle(reviews)
