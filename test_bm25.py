from corpus import corpus, jugements_pertinence  # Importation du corpus et de tous les jugements
from indexer import build_inverted_index
from bm25 import BM25
from evaluation import calcule_precision_at_k  # Importation de la fonction de précision

if __name__ == '__main__':
    # Analyse unique du corpus et génération de l'index complet
    corpus_metadata = build_inverted_index(corpus)
    
    # Initialisation du moteur de recherche avec les stats du corpus
    bm25 = BM25(corpus_metadata)
    
    # Liste des requêtes à tester (présentes dans vos jugements)
    queries = ["deep learning"]
    
    for query in queries:
        scores = bm25.get_scores(query)
        
        # Tri et affichage du classement complet
        ranked_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        print(f"\nQuery: '{query}'")
        print("BM25 top ranks:")
        for doc_id, score in ranked_results:
            text = corpus[doc_id]
            print(f"  doc {doc_id} score={score:.4f} text='{text}'")
        
        # Extraction de la liste ordonnée des IDs de documents pour l'évaluation
        classement_ids = [doc_id for doc_id, _ in ranked_results]
        
        # Calcul et affichage de la Précision@5 en utilisant les imports
        p5 = calcule_precision_at_k(classement_ids, query, jugements_pertinence, k=5)
        print(f"ÉVALUATION : Précision@5 pour '{query}' = {p5 * 100:.1f}%")
        print("-" * 50)