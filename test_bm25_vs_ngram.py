import indexer
from corpus import corpus
from indexer import build_inverted_index
from bm25 import BM25
from n_gram_tokenizer import NGramTokenizer
from tfidf import tfidf  # Import de votre fonction TF-IDF

if __name__ == '__main__':
    query = "deep learning"
    
    # On sauvegarde la fonction de tokenisation native (standard)
    standard_tokenize = indexer.tokenize 

    # ==========================================
    # CONFIGURATION 1 : BM25 SEUL
    # ==========================================
    print(f"--- DÉCOUPAGE DES DOCUMENTS (BM25 SEUL | MODE: WORD) ---")
    for i, doc in enumerate(corpus):
        print(f"Doc {i}: {indexer.tokenize(doc)}")
    print("-" * 60)
    
    meta_standard = build_inverted_index(corpus)
    bm25_standard = BM25(meta_standard)
    bm25_standard.tokenizer = standard_tokenize
    scores_standard = bm25_standard.get_scores(query)
    ranked_standard = sorted(enumerate(scores_standard), key=lambda x: x[1], reverse=True)

    # ==========================================
    # CONFIGURATION 2 : BM25 + N-GRAM
    # ==========================================
    ngram_tokenizer = NGramTokenizer(n=2, mode='word')
    indexer.tokenize = ngram_tokenizer  # Injection dynamique
    
    print(f"\n--- DÉCOUPAGE DES DOCUMENTS (BM25 + N-GRAM | N={ngram_tokenizer.n} | MODE: {ngram_tokenizer.mode.upper()}) ---")
    for i, doc in enumerate(corpus):
        print(f"Doc {i}: {indexer.tokenize(doc)}")
    print("-" * 60)
    
    meta_ngram = build_inverted_index(corpus)
    bm25_ngram = BM25(meta_ngram)
    bm25_ngram.tokenizer = ngram_tokenizer
    scores_ngram = bm25_ngram.get_scores(query)
    ranked_ngram = sorted(enumerate(scores_ngram), key=lambda x: x[1], reverse=True)

    # Restauration de l'indexer à son état d'origine
    indexer.tokenize = standard_tokenize

    # ==========================================
    # CONFIGURATION 3 : TF-IDF SEUL
    # ==========================================
    # Appel direct de votre fonction avec le corpus et la requête brute
    scores_tfidf = tfidf(corpus, query)
    ranked_tfidf = sorted(enumerate(scores_tfidf), key=lambda x: x[1], reverse=True)

    # ==========================================
    # AFFICHAGE COMPARATIF DES RÉSULTATS
    # ==========================================
    print(f"\nQuery: '{query}'")
    print(f"  -> Version Standard analysée comme : {standard_tokenize(query)}")
    print(f"  -> Version N-Gram analysée comme   : {ngram_tokenizer(query)}")

    print("\n==============================================")
    print("RÉSULTATS : TF-IDF")
    print("==============================================")
    for doc_id, score in ranked_tfidf:
        text = corpus[doc_id]
        print(f"  doc {doc_id} score={score:.4f} text='{text[:80]}...'")
    
    print("\n==============================================")
    print("RÉSULTATS : BM25")
    print("==============================================")
    for doc_id, score in ranked_standard:
        text = corpus[doc_id]
        print(f"  doc {doc_id} score={score:.4f} text='{text[:80]}...'")

    print("\n==============================================")
    print("RÉSULTATS : BM25 + N-GRAM")
    print("==============================================")
    for doc_id, score in ranked_ngram:
        text = corpus[doc_id]
        print(f"  doc {doc_id} score={score:.4f} text='{text[:80]}...'")

    