import string;
import re;
import random
import math
import numpy as np
from gensim.models import Word2Vec
import pickle as pkl
from itertools import *
import gensim
import tempfile
embed_d = 128
window_s = 5


def load_data(path="../dataset/paper_abstract.pkl", maxlen = None, n_words = 600000, sort_by_len = False):
    f = open(path, 'rb')
    content_set = pkl.load(f)
    f.close()

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    content_set_x, content_set_y = content_set

    content_set_x = remove_unk(content_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(content_set_x)
        content_set_x = [content_set_x[i] for i in sorted_index]

    return content_set_x


def word2vec():
    # generate word embedding file: word_embeddings.txt
    data = load_data()
    model = gensim.models.Word2Vec(data, vector_size = embed_d, window = window_s, min_count=0)
    model.wv.save_word2vec_format("../dataset/word_embed.txt")
    
word2vec()








