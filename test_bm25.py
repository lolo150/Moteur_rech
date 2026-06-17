from corpus import corpus
from indexer import build_inverted_index
from bm25 import BM25

if __name__ == '__main__':
    # 1. Analyse unique du corpus et génération de l'index complet
    corpus_metadata = build_inverted_index(corpus)
    
    # 2. Initialisation du moteur de recherche avec les stats du corpus
    bm25 = BM25(corpus_metadata)
    
    # 3. Exécution d'une requête
    query = "deep"
    scores = bm25.get_scores(query)
    
    # 4. Tri et affichage du classement complet
    ranked_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    print(f"Query: '{query}'\n")
    print("BM25 top ranks:")
    for doc_id, score in ranked_results:
        # On utilise le corpus initial uniquement pour l'affichage final du texte
        text = corpus[doc_id]
        print(f"  doc {doc_id} score={score:.4f} text='{text}'")



# def load_corpus_from_csv(path, text_col='product_name', max_docs=500):
#     print(f"Chargement de : {path}")
#     if not os.path.exists(path):
#         return None
#     texts = []
#     with open(path, newline='', encoding='utf-8', errors='ignore') as fh:
#         reader = csv.DictReader(fh, delimiter='\t')
#         for i, row in enumerate(reader):
#             if i >= max_docs:
#                 break
#             # Correction : Utilisation dynamique de text_col
#             text = row.get(text_col) or row.get('product_name') or ''
#             # print(text)
#             texts.append(text)
#     return texts
