"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-26 10:38:19
"""

import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, ppmi, cos_similarity
import numpy as np
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)
np.set_printoptions(precision=3)
print('covariance matrix:\n', C)
print('-' * 50)
print('PPMI\n', W)
