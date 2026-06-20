import re
from collections import defaultdict
from n_gram_tokenizer import NGramTokenizer

class NGramAutocomplete:
    def __init__(self, n=3):
        self.tokenizer = NGramTokenizer(n=n, mode='char')
        self.lexicon = set()
        self.char_index = defaultdict(set)
        
        # NOUVEAU : Index de bi-grammes de mots (mot_actuel -> mots_suivants)
        self.word_transitions = defaultdict(lambda: defaultdict(int))
        
    def fit(self, corpus):
        """Extrait les mots et apprend les transitions de mots (bi-grammes)."""
        for doc in corpus:
            words = re.findall(r"\w+", doc.lower())
            
            # 1. Apprentissage des mots isolés (Caractères)
            for word in words:
                if word not in self.lexicon:
                    self.lexicon.add(word)
                    sub_sequences = self.tokenizer(word)
                    for seq in sub_sequences:
                        self.char_index[seq].add(word)
            
            # 2. NOUVEAU : Apprentissage des paires de mots successifs
            for i in range(len(words) - 1):
                word_a = words[i]
                word_b = words[i+1]
                self.word_transitions[word_a][word_b] += 1
                        
    def suggest(self, prefix, limit=5):
        """Retourne des suggestions de mots ou de complétions de phrases."""
        raw_prefix = prefix.lower()
        
        # CAS 1 : L'utilisateur a tapé un mot et a mis un espace (ex: "deep ")
        # On cherche à prédire le MOT SUIVANT
        if raw_prefix.endswith(" "):
            words = re.findall(r"\w+", raw_prefix)
            if not words:
                return []
            last_word = words[-1]
            
            if last_word in self.word_transitions:
                # On récupère les mots suivants les plus fréquents
                next_words = self.word_transitions[last_word]
                sorted_next = sorted(next_words.keys(), key=lambda x: next_words[x], reverse=True)
                
                # On recrée l'expression complète (ex: "deep learning")
                return [f"{last_word} {next_word}" for next_word in sorted_next][:limit]
            return []

        # CAS 2 : L'utilisateur est en train de taper un mot (ex: "dee")
        prefix = raw_prefix.strip()
        if not prefix:
            return []
            
        exact_matches = [w for w in self.lexicon if w.startswith(prefix)]
        exact_matches = sorted(exact_matches, key=len)
        
        if len(exact_matches) >= limit:
            return exact_matches[:limit]
            
        prefix_grams = self.tokenizer(prefix)
        scores = defaultdict(int)
        for gram in prefix_grams:
            if gram in self.char_index:
                for candidate_word in self.char_index[gram]:
                    if len(candidate_word) >= len(prefix):
                        scores[candidate_word] += 1
                    
        ngram_matches = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        results = exact_matches + [w for w in ngram_matches if w not in exact_matches]
        return results[:limit]
    




def distance_levenshtein(mot1, mot2):
    """Calcule la distance de Levenshtein exacte entre deux mots (Programmation dynamique)."""
    m, n = len(mot1), len(mot2)
    # Création de la matrice de mémorisation
    matrice = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1): matrice[i][0] = i
    for j in range(n + 1): matrice[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if mot1[i-1] == mot2[j-1]:
                cost = 0
            else:
                cost = 1
            matrice[i][j] = min(
                matrice[i-1][j] + 1,      # Suppression
                matrice[i][j-1] + 1,      # Insertion
                matrice[i-1][j-1] + cost  # Substitution
            )
    return matrice[m][n]

class LevenshteinAutocomplete:
    def __init__(self, max_distance=2):
        self.lexique = set()
        self.max_distance = max_distance

    def fit(self, dataset):
        """Construit le dictionnaire de mots uniques à partir du corpus."""
        for doc in dataset:
            words = re.findall(r"\w+", doc.lower())
            self.lexique.update(words)

    def suggest(self, current_word, limit=5):
        """Propose des corrections basées sur la distance minimale."""
        word = current_word.lower().strip()
        if not word:
            return []
        
        # Si le mot exact existe, pas besoin de corriger
        if word in self.lexique:
            return [word]
            
        suggestions = []
        for candidat in self.lexique:
            dist = distance_levenshtein(word, candidat)
            if dist <= self.max_distance:
                suggestions.append((candidat, dist))
                
        # Tri par distance la plus petite, puis par ordre alphabétique
        suggestions.sort(key=lambda x: (x[1], x[0]))
        return [sug[0] for sug in suggestions[:limit]]    