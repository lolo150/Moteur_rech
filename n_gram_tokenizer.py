import re

class NGramTokenizer:
    def __init__(self, n=2, mode='word'):
        """
        n: taille des n-grams
        mode: 'word' pour les expressions, 'char' pour les caractères
        """
        self.n = n
        self.mode = mode

    def __call__(self, text):
        if not isinstance(text, str):
            return []
        
        text = text.lower()
        
        if self.mode == 'word':
            # 1. Mode Mots : Extraction des unigrams
            base_tokens = re.findall(r"\w+", text)
            if self.n <= 1 or len(base_tokens) < self.n:
                return base_tokens
                
            # 2. Génération des n-grams de mots (liés par '_')
            ngram_tokens = []
            for i in range(len(base_tokens) - self.n + 1):
                ngram = "_".join(base_tokens[i:i + self.n])
                ngram_tokens.append(ngram)
                
            return base_tokens + ngram_tokens

        elif self.mode == 'char':
            # 1. Mode Caractères : Nettoyage des espaces et extraction des unigrams
            text_cleaned = re.sub(r"\s+", " ", text)
            base_tokens = list(text_cleaned)
            if self.n <= 1 or len(base_tokens) < self.n:
                return base_tokens
                
            # 2. Génération des n-grams de caractères (ex: 'dee', 'eep')
            ngram_tokens = []
            for i in range(len(base_tokens) - self.n + 1):
                ngram = "".join(base_tokens[i:i + self.n])
                ngram_tokens.append(ngram)
                
            return base_tokens + ngram_tokens
            
        else:
            raise ValueError("Le mode doit être 'word' ou 'char'")