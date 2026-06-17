import os
import csv
import math
from bm25 import BM25
from corpus import corpus

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




def make_saturation_corpus(term='apple'):
    """Create a small corpus that demonstrates saturation and length effects.

    Documents:
      0: short, one occurrence
      1: medium, few occurrences
      2: very long, many occurrences (will 'saturate')
      3: long but only one occurrence (tests length normalization)
      4: medium with some occurrences spread among filler
      5: unrelated doc (noise)
    """
    filler = ' '.join(['filler'] * 200)
    corpus = []

    # doc 0: short, 1 occurrence
    corpus.append(f"{term}")

    # doc 1: medium, repeated 5 times
    corpus.append((' ' + term) * 5 + ' ' + 'mid ' * 20)

    # doc 2: very long, repeated 80 times (lots of term occurrences)
    corpus.append((term + ' ') * 80 + ' ' + filler)

    # doc 3: very long but only one occurrence
    corpus.append(filler + ' ' + term)

    # doc 4: medium with some occurrences spread
    corpus.append((' ' + term) * 10 + ' ' + ' '.join(['other'] * 50))

    # doc 5: unrelated
    corpus.append(' '.join(['other'] * 100))

    return corpus







def run_demo():
    # try to load corpus from data file
    
    current_corpus = corpus

    query = 'Deep learning'
    query_tokens = query.lower().split() # ['deep', 'learning']

    print(f"Corpus size: {len(current_corpus)}")

    tokenized = [doc.lower().replace('.', '').split() for doc in current_corpus] # Nettoyage rapide de la ponctuation
    for i, doc in enumerate(tokenized):
        length = len(doc)
        # Compte pour chaque token individuellement
        counts = [f"{token}: {doc.count(token)}" for token in query_tokens]
        counts_str = " | ".join(counts)
        
        # Affichage du résultat pour chaque document
        print(f"  Doc {i}: len={length} -> {counts_str}")
    # BM25
    bm25 = BM25(current_corpus)
    bm_scores = bm25.get_scores(query)

    # TF-IDF (sklearn if available)
    TfidfVec, cos_sim = try_import_sklearn()
    if TfidfVec is not None:
        print('\nUsing scikit-learn TF-IDF (default parameters)')
        vect = TfidfVec()
        X = vect.fit_transform(current_corpus)
        qv = vect.transform([query])
        tf_scores = cos_sim(qv, X).flatten().tolist()
    else:
        print('\nUsing TF-IDF from scratch (no sklearn)')
        tf_scores = tfidf(current_corpus, query)

    # show rankings
    ranked_bm = sorted(enumerate(bm_scores), key=lambda x: x[1], reverse=True)
    ranked_tf = sorted(enumerate(tf_scores), key=lambda x: x[1], reverse=True)

    print('\nQuery:', query)
    print('\nTF-IDF top ranks:')
    for i, s in ranked_tf:
        text = current_corpus[i] or ''
        safe = text.replace("'", "\\'")[:120]
        print(f"  doc {i} score={s:.4f} text='{safe}'")

    print('\nBM25 top ranks:')
    for i, s in ranked_bm:
        text = current_corpus[i] or ''
        safe = text.replace("'", "\\'")[:120]
        print(f"  doc {i} score={s:.4f} text='{safe}'")
    

if __name__ == '__main__':
    run_demo()
