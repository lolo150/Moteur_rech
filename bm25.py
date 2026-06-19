import math
import re

class BM25:
    def __init__(self, corpus_stats, k1=1.5, b=0.75, tokenizer=None):
        """
        Initialise le moteur BM25.
        
        :param corpus_stats: Dictionnaire contenant 'index', 'doc_lens', 'N' et 'avgdl'
        :param k1: Paramètre de saturation du terme de fréquence (default: 1.5)
        :param b: Paramètre de normalisation de la longueur des documents (default: 0.75)
        :param tokenizer: Callable optionnel pour parser la requête
        """
        self.k1 = float(k1)
        self.b = float(b)
        
        # Récupération des statistiques globales précalculées
        self.index = corpus_stats["index"]
        self.doc_lens = corpus_stats["doc_lens"]
        self.N = int(corpus_stats["N"])
        self.avgdl = float(corpus_stats["avgdl"])
        
        # Injection ou repli sur le tokenizer par défaut (regex unigramme)
        self.tokenizer = tokenizer if tokenizer is not None else lambda text: re.findall(r"\w+", text.lower())
        
        # Précalcul des scores IDF (Formule standard de Robertson-Spärck Jones)
        self.idf = {}
        self._precompute_idf()

    def _precompute_idf(self):
        """Précalcule l'Inverse Document Frequency pour chaque terme du lexique."""
        for term, posting_list in self.index.items():
            df = len(posting_list)
            # Ajout du +1.0 pour lisser et éviter les valeurs d'IDF négatives
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query):
        """
        Calcule les scores de similarité BM25 pour l'ensemble des documents.
        
        :param query: Chaîne de caractères de la requête utilisateur
        :return: Liste de flottants représentant le score de chaque doc_id
        """
        q_terms = self.tokenizer(query)
        scores = [0.0] * self.N
        
        # Alias locaux pour optimiser l'accès en boucle fermée
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl
        doc_lens = self.doc_lens
        
        for term in q_terms:
            if term not in self.index:
                continue
                
            idf = self.idf.get(term, 0.0)
            posting_list = self.index[term]
            
            # Normalisation du parcours de la posting list (compatible Dict et List)
            items = posting_list.items() if hasattr(posting_list, 'items') else posting_list
            
            for doc_id, term_freq in items:
                dl = doc_lens[doc_id]
                
                # Formule BM25 : IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (dl / avgdl)))
                denom = term_freq + k1 * (1.0 - b + b * (dl / avgdl))
                scores[doc_id] += idf * (term_freq * (k1 + 1.0)) / denom
                
        return scores
    








