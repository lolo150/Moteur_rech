def try_import_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        return TfidfVectorizer, cosine_similarity
    except Exception:
        return None, None


def tfidf(corpus, query):
    """TF-IDF avec normalisation classique (tf / de la longueur du doc)."""
    from collections import Counter
    import math

    tokenized = [doc.lower().split() for doc in corpus]
    q_tokens = query.lower().split()

    # document frequency
    import collections
    df = collections.Counter()
    for doc in tokenized:
        for t in set(doc):
            df[t] += 1

    N = len(corpus)
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    def tfidf_vec(tokens):
        tf = Counter(tokens)
        doc_len = len(tokens) # Nombre total de mots
        
        # CHANGEMENT ICI : tf normalisé = f / doc_len
        return {t: (f / doc_len) * idf.get(t, 0.0) for t, f in tf.items()}

    qv = tfidf_vec(q_tokens)

    def dot(a, b):
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    scores = []
    for doc in tokenized:
        dv = tfidf_vec(doc)
        # CHANGEMENT ICI : Simple produit scalaire, plus de division par les normes
        scores.append(dot(dv, qv))
    return scores

