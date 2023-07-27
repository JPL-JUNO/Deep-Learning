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
    M = np.zeros_like(C, dtype=np.float64)
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


def create_contexts_targets(corpus: list, window_size: int = 1) -> tuple[ndarray, ndarray]:
    """实现生成上下文和目标词的函数

    Parameters
    ----------
    corpus : list
        语料库
    window_size : int, optional
        窗口大小, by default 1

    Returns
    -------
    tuple[ndarray, ndarray]
        上下文和目标词
    >>> corpus = [0 1 2 3 4 1 5 6]
    >>> create_contexts_targets(corpus, window_size=1)
    ... (array([[0, 2],
    ...         [1, 3],
    ...         [2, 4],
    ...         [3, 1],
    ...         [4, 5],
    ...         [1, 6]]),
    ...  array([1, 2, 3, 4, 1, 5]))
    """
    target = corpus[window_size:-window_size]
    contexts = []
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus: ndarray, vocab_size: int) -> ndarray:
    """转换为one-hot表示

    Parameters
    ----------
    corpus : ndarray
        单词 ID 列表，表示上下文
    vocab_size : int
        词汇个数

    Returns
    -------
    ndarray
        one-hot表示
    >>> convert_one_hot(contexts, 7)
    ... array([[[1, 0, 0, 0, 0, 0, 0],
    ...         [0, 0, 1, 0, 0, 0, 0]],
    ...
    ...        [[0, 1, 0, 0, 0, 0, 0],
    ...         [0, 0, 0, 1, 0, 0, 0]],
    ...
    ...        [[0, 0, 1, 0, 0, 0, 0],
    ...         [0, 0, 0, 0, 1, 0, 0]],
    ...
    ...        [[0, 0, 0, 1, 0, 0, 0],
    ...          [0, 1, 0, 0, 0, 0, 0]],
    ...     
    ...        [[0, 0, 0, 0, 1, 0, 0],
    ...          [0, 0, 0, 0, 0, 1, 0]],
    ...     
    ...        [[0, 1, 0, 0, 0, 0, 0],
    ...          [0, 0, 0, 0, 0, 0, 1]]])
    """
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    return one_hot


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    print(word_to_id)
    print(C)
    ppmi(C, verbose=True)
