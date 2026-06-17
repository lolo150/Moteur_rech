import re
from collections import defaultdict

def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())

def build_inverted_index(corpus):
    """Analyse le corpus et extrait les structures de données nécessaires."""
    index = defaultdict(lambda: defaultdict(int))
    doc_lens = []
    
    for doc_id, doc in enumerate(corpus):
        tokens = tokenize(doc)
        doc_lens.append(len(tokens))
        
        for token in tokens:
            index[token][doc_id] += 1
            
    # Statistiques globales du corpus
    N = len(corpus)
    avgdl = sum(doc_lens) / N if N > 0 else 0.0
    
    return {
        "index": index,
        "doc_lens": doc_lens,
        "N": N,
        "avgdl": avgdl
    }