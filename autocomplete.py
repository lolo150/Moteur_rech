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