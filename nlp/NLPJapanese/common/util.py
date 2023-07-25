"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-25 10:20:19
"""
import numpy as np
from numpy import ndarray


def clip_grads():
    pass


def preprocess(text: str):
    words = text.lower().replace('.', ' .').split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[word] for word in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus: ndarray, vocab_size: int, window_size: int = 1) -> ndarray:
    """计算语料库中各个单词间的共现矩阵

    Parameters
    ----------
    corpus : ndarray
        单词 ID 的列表，由 word_to_id 编制 corpus 的列表，将 corpus 中的每个单词，按照 word_to_id 编制成数组
    vocab_size : int
        word_to_id 的长度，即词汇的个数
    window_size : int, optional
        窗口大小，将上下文的大小（即周围的单词有多少个）称为窗口大小（window size）, 
        by default 1

    Returns
    -------
    ndarray
        共现矩阵（方阵），维度等于 vocab_size x vocab_size
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id: dict, id_to_word: dict,
                 word_matrix: ndarray, top: int = 5):
    # 取出查询词
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    print('\b[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 计算余弦相似度
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    cnt = 0
    for i in (-1 * similarity).argsort():
        pass
    pass


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(create_co_matrix(corpus, 7))
