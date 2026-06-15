import math
import re
from collections import Counter, defaultdict


def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())


class BM25:
    """A small, dependency-free implementation of Okapi BM25.

    Usage:
        bm25 = BM25(corpus)
        scores = bm25.get_scores(query)
        top = bm25.get_top_n(query, n=5)
    """

    def __init__(self, corpus, k1=1.5, b=0.75, tokenizer=tokenize):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)

        # tokenized documents
        self.doc_tokens = [self.tokenizer(doc) for doc in corpus]
        self.N = len(self.doc_tokens)

        # term frequencies per document (list of Counter)
        self.tfs = [Counter(d) for d in self.doc_tokens]

        # document lengths
        self.doc_lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0.0

        # document frequencies: term -> number of docs containing term
        self.doc_freqs = defaultdict(int)
        for tf in self.tfs:
            for term in tf.keys():
                self.doc_freqs[term] += 1

        # precompute idf for each term using BM25 idf formula with smooth
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # standard BM25 idf (with +1 smoothing inside log)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def get_score_for_doc(self, query_terms, index):
        score = 0.0
        tf = self.tfs[index]
        dl = self.doc_lens[index]
        for term in query_terms:
            if term not in tf:
                continue
            term_freq = tf[term]
            idf = self.idf.get(term, 0.0)
            denom = term_freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (term_freq * (self.k1 + 1)) / denom
        return score

    def get_scores(self, query):
        q_terms = self.tokenizer(query)
        return [self.get_score_for_doc(q_terms, i) for i in range(self.N)]

    def get_top_n(self, query, n=5):
        scores = self.get_scores(query)
        ranked_ix = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:n]
        return [self.corpus[i] for i in ranked_ix], [scores[i] for i in ranked_ix]
