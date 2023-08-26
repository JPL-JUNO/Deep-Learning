"""
@Title: 词关系可视化
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-26 14:52:07
@Description: nlpia 这个包有问题，版本没跟上导致无法运行
"""


import os
# 版本不兼容
from nlpia.loaders import get_data
from gensim.models.word2vec import KeyedVectors
wv = KeyedVectors.load_word2vec_format("", binary=True)
wv = get_data("word2vec")
assert len(wv.vocab) == 3_000_000

import pandas as pd
vocab = pd.Series(wv.vocab)


import numpy as np
np.linalg.norm(wv["Illinois"] - wv["Illini"])


cos_similarity = np.dot(wv["Illinois"], wv["Illini"]) / \
    (np.linalg.norm(wv["Illinois"]) * np.linalg.norm(wv["Illini"]))

cities = get_data("cities")

us = cities[(cities["country_code"] == "US") &
            (cities["admin1_code"].notnull())].copy()
states = pd.read_csv(
    "http://www.fonz.net/blog/wp-content/uploads/2008/04/states.csv")
states = dict(zip(states["Abbreviated"], states["States"]))
us["city"] = us["name"].copy()
us["st"] = us["admin1_code"].copy()
us["state"] = us["st"].map(states)
us[us.columns[-3:]].head()

vocab = pd.concat([us["city"], us["st"], us["state"]])
# 存疑
# vocab = pd.np.concatenate([us["city"], us["st"], us["state"]])
vocab = np.array([word for word in vocab if word in wv.wv])

# 通过州向量来增强城市向量（因为存在不同的州有相同的城市名）
city_plus_state = []
for c, state, st in zip(us["city"], us["state"], us["st"]):
    if c not in vocab:
        continue
    row = []
    if state in vocab:
        row.extend(wv[c] + wv[state])
    else:
        row.extend(wv[c] + wv[st])
    city_plus_state.append(row)
us_300 = pd.DataFrame(city_plus_state)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
us_300D = get_data('cities_us_wordvectors')
us_2D = pca.fit_transform(us_300D.iloc[:, :300])


import seaborn
import matplotlib.pyplot as plt
# 版本不兼容
from nlpia.plots import offline_plotly_scatter_bubble
df = get_data("cities_us_wordvectors_pca2_meta")
html = offline_plotly_scatter_bubble(
    df.sort_values("population", ascending=False)[
        :350].copy().sort_values("population"),
    filename='plotly_scatter_bubble.html',
    x='x', y='y', size_col="population", text_col="name", category_col="timezone",
    xscale=None, yscale=None,
    layout={}, marker={'sizeof': 3_000}
)
