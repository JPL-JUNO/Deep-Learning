"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-26 09:11:36
"""

import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similar
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
most_similar('you', word_to_id, id_to_word, C, top=5)

print('降 维 的 方 法 有 很 多， 这 里 我 们 使 用 奇 异 值 分 解'.replace(' ', ''))
