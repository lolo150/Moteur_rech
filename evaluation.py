def calcule_precision_at_k(classement_ids, requete, jugements, k=5):
    """Calcule la précision@K pour une requête à partir des jugements importés."""
    if requete not in jugements:
        print(f"⚠️ Aucun jugement disponible pour la requête : '{requete}'")
        return 0.0
    
    jugements_req = jugements[requete]
    top_k_docs = classement_ids[:k]
    
    if not top_k_docs:
        return 0.0
        
    docs_pertinents = sum(1 for doc_id in top_k_docs if jugements_req.get(doc_id, 0) == 1)
    return docs_pertinents / len(top_k_docs)