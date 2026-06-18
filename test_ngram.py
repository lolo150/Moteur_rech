import indexer
from corpus import corpus
from indexer import build_inverted_index
from bm25 import BM25
from n_gram_tokenizer import NGramTokenizer  # Nom du fichier où est stocké votre NGramTokenizer

if __name__ == '__main__':
    # 1. Configuration du Tokenizer (Modifiez ici : mode='word' ou mode='char')
    # Exemple ici en mode WORD avec Bigrames (n=2)
    ngram_tokenizer = NGramTokenizer(n=3, mode='char')
    
    # Injection dynamique dans le module indexer
    indexer.tokenize = ngram_tokenizer

    # 2. AFFICHAGE DU DÉCOUPAGE DE CHAQUE DOCUMENT
    print(f"--- DÉCOUPAGE DES DOCUMENTS (N={ngram_tokenizer.n} | MODE: {ngram_tokenizer.mode.upper()}) ---")
    for i, doc in enumerate(corpus):
        # On appelle le tokenizer directement pour voir les tokens générés
        tokens_générés = ngram_tokenizer(doc)
        print(f"Doc {i}: {tokens_générés}")
    print("-" * 50)

    # 3. Construction de l'index avec le tokenizer injecté
    corpus_metadata = build_inverted_index(corpus)
    
    # 4. Initialisation et configuration de BM25
    bm25 = BM25(corpus_metadata)
    bm25.tokenizer = ngram_tokenizer  # On s'assure que BM25 utilise le même tokenizer pour la requête
    
    # 5. Exécution de la requête
    query = "deeep"
    scores = bm25.get_scores(query)
    
    # 6. Tri et affichage du classement
    ranked_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    print(f"\nQuery: '{query}' (Analysée comme : {ngram_tokenizer(query)})\n")
    print("BM25 + N-Gram top ranks:")
    for doc_id, score in ranked_results:
        text = corpus[doc_id]
        print(f"  doc {doc_id} score={score:.4f} text='{text[:80]}...'")

        