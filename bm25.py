import math
import re

class BM25:
    def __init__(self, corpus_stats, k1=1.5, b=0.75):
        self.k1 = float(k1)
        self.b = float(b)
        
        # Récupération des données précalculées
        self.index = corpus_stats["index"]
        self.doc_lens = corpus_stats["doc_lens"]
        self.N = corpus_stats["N"]
        self.avgdl = corpus_stats["avgdl"]
        
        # Tokenizer interne pour traiter la requête de la même manière
        self.tokenizer = lambda text: re.findall(r"\w+", text.lower())
        
        # Précalcul des IDF
        self.idf = {}
        for term, posting_list in self.index.items():
            df = len(posting_list)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query):
        """Calcule les scores uniquement pour les documents pertinents."""
        q_terms = self.tokenizer(query)
        scores = [0.0] * self.N
        
        for term in q_terms:
            if term not in self.index:
                continue
                
            idf = self.idf.get(term, 0.0)
            posting_list = self.index[term]
            
            for doc_id, term_freq in posting_list.items():
                dl = self.doc_lens[doc_id]
                denom = term_freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * (term_freq * (self.k1 + 1)) / denom
                
        return scores