import os
from pprint import pprint
import math

from bm25 import BM25


def try_import_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        return TfidfVectorizer, cosine_similarity
    except Exception:
        return None, None


def simple_tfidf_cosine(corpus, query):
    # simple TF-IDF with python and math (no sklearn). Returns cosine scores.
    from collections import Counter
    import math

    tokenized = [doc.lower().split() for doc in corpus]
    q_tokens = query.lower().split()

    # build df
    import collections
    df = collections.Counter()
    for doc in tokenized:
        for t in set(doc):
            df[t] += 1

    N = len(corpus)
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    def tfidf_vec(tokens):
        tf = Counter(tokens)
        vec = {}
        for t, f in tf.items():
            vec[t] = f * idf.get(t, 0.0)
        return vec

    qv = tfidf_vec(q_tokens)

    def dot(a, b):
        s = 0.0
        for k, v in a.items():
            s += v * b.get(k, 0.0)
        return s

    def norm(a):
        return math.sqrt(sum(v * v for v in a.values()))

    scores = []
    for doc in tokenized:
        dv = tfidf_vec(doc)
        denom = norm(dv) * norm(qv)
        score = dot(dv, qv) / denom if denom > 0 else 0.0
        scores.append(score)
    return scores


def run_example(corpus, queries):
    print(f"Corpus size: {len(corpus)}")
    bm25 = BM25(corpus)

    TfidfVec, cos_sim = try_import_sklearn()
    use_sklearn = TfidfVec is not None

    if use_sklearn:
        print("Using scikit-learn TF-IDF for comparison")
        vect = TfidfVec()
        X = vect.fit_transform(corpus)

    for q in queries:
        print('\nQuery:', q)
        bm_scores = bm25.get_scores(q)
        # show top 5 docs for BM25
        ranked_bm = sorted(enumerate(bm_scores), key=lambda x: x[1], reverse=True)[:5]
        print('\nBM25 top 5:')
        for i, s in ranked_bm:
            print(f'  doc {i} score={s:.4f} text={corpus[i][:100]!r}')

        if use_sklearn:
            qv = vect.transform([q])
            sims = cos_sim(qv, X).flatten()
            ranked_tf = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:5]
        else:
            sims = simple_tfidf_cosine(corpus, q)
            ranked_tf = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:5]

        print('\nTF-IDF top 5:')
        for i, s in ranked_tf:
            print(f'  doc {i} score={s:.4f} text={corpus[i][:100]!r}')


def load_small_corpus_from_csv(path, text_col='product_name', n=200):
    try:
        import pandas as pd
    except Exception:
        return None
    try:
        df = pd.read_csv(path, usecols=[text_col])
    except Exception:
        return None
    texts = df[text_col].fillna('').astype(str).tolist()
    return texts[:n]


if __name__ == '__main__':
    # try to load sample CSV in repo
    sample_csv = os.path.join(os.path.dirname(__file__), 'flipkart_com-ecommerce_sample.csv')
    corpus = None
    if os.path.exists(sample_csv):
        corpus = load_small_corpus_from_csv(sample_csv, text_col='product_name', n=300)
    if corpus is None:
        # fallback tiny corpus
        corpus = [
            'red running shoes for men',
            'women tennis shoes white',
            'bluetooth wireless headphones',
            'wireless mouse for laptop',
            'men leather belt brown',
            'water bottle stainless steel',
            'cotton t-shirt men black',
            'ladies handbag faux leather',
            'kids toy car remote control',
            'office chair ergonomic adjustable',
        ]

    queries = ['wireless headphones', 'running shoes', 'leather belt']
    run_example(corpus, queries)
