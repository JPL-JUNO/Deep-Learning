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
                 word_matrix: ndarray, top: int = 5) -> None:
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
        # 如果是查询词，相似度是 1，直接跳过
        if id_to_word[i] == query:
            continue
        print('%s: %s' % (id_to_word[i], similarity[i]))
        cnt += 1
        if cnt >= top:
            return


def ppmi(C: ndarray, verbose: bool = False, eps: float = 1e-8) -> ndarray:
    """计算正点互信息 positive pointwise mutual information

    Parameters
    ----------
    C : ndarray
        单词之间的共现矩阵
    verbose : bool, optional
        决定是否输出运行情况的标志。当处理大语料库时，设置 `verbose=True`，可以用于确认运行情况。, by default False
    eps : float, optional
        一个极小值，用于确保 log 中不会出现 0, by default 1e-8

    Returns
    -------
    ndarray
        与共现矩阵 `C` 相同大小的修正点互信息
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)  # 语料库中单词的数量
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]  # 没搞错的话，C 应该是个方阵
    cnt = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)
            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print('%.1f%% done' % (100 * cnt / total))
    return M


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(create_co_matrix(corpus, 7))
