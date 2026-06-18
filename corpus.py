# default inline corpus if data file missing
corpus = [
    "Deep learning. Deep learning",
    "Le deep learning est une technologie intéressante. Dans ce grand article, nous allons explorer en détail le deep learning sous tous ses angles...",
    "Apprendre le learning learning learning learning.",
    "J'adore l'apprentissage profond et le deep learning.",
    "Dans cet article sur l'intelligence artificielle, nous aborderons le deep puis plus loin le learning."
]


# Vérité terrain pour chaque requête spécifique
jugements_pertinence = {
    "deep learning": {
        0: 1,  # Doc 0 (Idéal, court et dense) -> Oui
        1: 1,  # Doc 1 (Bavard mais traite du sujet) -> Oui
        2: 0,  # Doc 2 (Spam de 'learning') -> Non
        3: 1   # Doc 3 (Contient l'idée malgré 'deeep') -> Oui
    },
    "learning": {
        0: 1,  # Doc 0 -> Oui
        1: 1,  # Doc 1 -> Oui
        2: 0,  # Doc 2 (Spam / Répétition abusive) -> Non (0 pour pénaliser le spam)
        3: 1   # Doc 3 -> Oui
    },
    "deeep": {
            0: 1,  # Pertinent (contient 'deep')
            1: 1,  # Pertinent (contient 'deep')
            2: 0,  # Non pertinent
            3: 1   # Pertinent (contient 'deep')
        }
}