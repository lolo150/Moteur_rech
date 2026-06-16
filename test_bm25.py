import os
import csv
import re
# Note : Assurez-vous d'utiliser une bibliothèque standard comme 'rank_bm25'
# ou que votre classe BM25 locale accepte ce format.
from bm25 import BM25 

def load_corpus_from_csv(path, text_col='product_name', max_docs=500):
    print(f"Chargement de : {path}")
    if not os.path.exists(path):
        return None
    texts = []
    with open(path, newline='', encoding='utf-8', errors='ignore') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for i, row in enumerate(reader):
            if i >= max_docs:
                break
            # Correction : Utilisation dynamique de text_col
            text = row.get(text_col) or row.get('product_name') or ''
            # print(text)
            texts.append(text)
    return texts

def run_demo():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'product(1).csv')
    current_corpus = load_corpus_from_csv(data_path, text_col='product_description', max_docs=500)
    # Correction : Gestion si le fichier est introuvable
    if not current_corpus:
        print(f"Erreur : Impossible de charger le fichier {data_path}")
        return

    print(f"Loaded corpus from {data_path} (n={len(current_corpus)})")

    query = 'sleep'
    query_tokens = query.lower().split() 

    # Tokenisation propre du corpus
    tokenized_corpus = [
        re.sub(r'[^\w\s]', '', doc.lower()).split() 
        for doc in current_corpus
    ]    
    
    # Affichage des comptes de tokens
    for i, doc_tokens in enumerate(tokenized_corpus):
        length = len(doc_tokens)
        counts = [f"{token}: {doc_tokens.count(token)}" for token in query_tokens]
        counts_str = " | ".join(counts)
        print(f"  Doc {i}: len={length} -> {counts_str}")
    
    # Correction BM25 : Initialisation et requête avec le corpus tokenisé
    bm25 = BM25(current_corpus)
    bm_scores = bm25.get_scores(query)

    # Tri des classements
    ranked_bm = sorted(enumerate(bm_scores), key=lambda x: x[1], reverse=True)

    print('\nQuery:', query)
    print('\nBM25 top ranks:')
    for i, s in ranked_bm[:10]: # Limité au top 10 pour plus de clarté
        text = current_corpus[i] or ''
        safe = text.replace("'", "\\'")[:120]
        print(f"  doc {i} score={s:.4f} text='{safe}'")

if __name__ == '__main__':
    run_demo()