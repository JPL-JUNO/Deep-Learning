"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-26 14:36:49
@Description: 
"""

MODEL_PATH = None
import gensim
if gensim.__version__ < '3.2.0':
    from gensim.models.wrappers.fasttext import FastText
else:
    from gensim.models.fasttext import FastText
ft_model = FastText.load_fasttext_format(model_file=MODEL_PATH)
ft_model.most_similar("phone")
