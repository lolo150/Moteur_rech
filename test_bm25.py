import math
from bm25 import BM25


def test_bm25_basic():
    corpus = [
        'the quick brown fox',
        'jumped over the lazy dog',
        'the fox',
    ]
    bm = BM25(corpus)
    scores = bm.get_scores('fox')
    # the documents containing 'fox' should have higher score than others
    print(scores)
    # print(scores[2] > 0)
    # print(scores[1] == 0)


test_bm25_basic()    
