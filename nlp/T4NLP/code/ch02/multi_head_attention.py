"""
@Title: Multi-head attention
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-28 10:53:22
@Description: 
"""

import numpy as np
from scipy.special import softmax

# Step 1: Represent the input
x = np.array([[1., 0., 1., 0.],
              [0., 2., 0., 2.],
              [1., 1., 1., 1.]])

# Step 2: Initializing the weight matrices
w_query = np.array([[1, 0, 1],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1]])
w_key = np.array([[0, 0, 1],
                  [1, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0]])
w_value = np.array([[0, 2, 0],
                    [0, 3, 0],
                    [1, 0, 3],
                    [1, 1, 0]])

# Step 3: Matrix multiplication to obtain Q, K, and V
Q = np.matmul(x, w_query)
K = np.matmul(x, w_key)
V = np.matmul(x, w_value)

# Step 4: Scaled attention scores
k_d = 1  # square root of k_d=3 rounded down to 1
attention_scores = (Q @ K.transpose()) / k_d

# Step 5: Scaled softmax attention scores for each vector
# attention_scores[0] = softmax(attention_scores[0])
# attention_scores[1] = softmax(attention_scores[1])
# attention_scores[2] = softmax(attention_scores[2])
attention_scores = softmax(attention_scores, axis=1)

# Step 6: The final attention representations
attention1 = attention_scores[0][0] * V[0]
attention2 = attention_scores[0][1] * V[1]
attention3 = attention_scores[0][2] * V[2]
# attention_scores[0].reshape(-1, 1) * V # 这是一个更加简洁的写法

# Step 7: Summing up the results
attention_input1 = attention1 + attention2 + attention3
attention_input2 = (attention_scores[1].reshape(-1, 1) * V).sum(axis=0)
attention_input3 = (attention_scores[2].reshape(-1, 1) * V).sum(axis=0)
# 下面的这个结果和上面的分次写是一样的，但是不太好理解，得想一会
attention_input = (attention_scores.reshape(-1, 3, 1) * V).sum(axis=1)

# Step 8: Steps 1 to 7 for all the inputs
attention_head1 = np.random.random((3, 64))

# Step 9: The output of the heads of the attention sublayer
z0h1 = np.random.random((3, 64))
z1h2 = np.random.random((3, 64))
z2h3 = np.random.random((3, 64))
z3h4 = np.random.random((3, 64))
z4h5 = np.random.random((3, 64))
z5h6 = np.random.random((3, 64))
z6h7 = np.random.random((3, 64))
z7h8 = np.random.random((3, 64))

# Step 10: Concatenation of the output of the heads
output_attention = np.hstack((z0h1, z1h2, z2h3, z3h4, z4h5, z5h6, z6h7, z7h8))
