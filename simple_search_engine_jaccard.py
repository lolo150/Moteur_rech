import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# 1) Charger les fichiers product.csv et query.csv
# ============================================================

def find_file(filename):
    """
    Cherche un fichier dans le dossier courant puis dans /mnt/data.
    Accepte aussi les noms avec (1), par exemple product(1).csv.
    """
    candidates = [
        Path(filename),
        Path("/mnt/data") / filename,
    ]

    # Cas où le fichier s'appelle product(1).csv ou query(1).csv
    if filename == "product.csv":
        candidates += [
            Path("product(1).csv"),
            Path("/mnt/data") / "product(1).csv",
        ]

    if filename == "query.csv":
        candidates += [
            Path("query(1).csv"),
            Path("/mnt/data") / "query(1).csv",
        ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Fichier introuvable : {filename}")


def load_data():
    product_path = find_file("product.csv")
    query_path = find_file("query.csv")

    product_df = pd.read_csv(product_path, sep="\t")
    query_df = pd.read_csv(query_path, sep="\t")

    # On utilise seulement les 3 premières requêtes
    query_df_3 = query_df.head(3).copy()

    return product_df, query_df_3


# ============================================================
# 2) Nettoyer le texte
# ============================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "with", "you", "your", "can", "will", "its", "have", "has", "had",
    "but", "not", "into", "more", "than", "such"
}


def preprocess(text):
    """
    Transforme un texte en liste de mots propres.

    Exemple :
    "Smart Coffee Table!" devient ["smart", "coffee", "table"]
    """
    if pd.isna(text):
        return []

    text = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+", text)

    clean_tokens = [
        token
        for token in tokens
        if len(token) > 1 and token not in STOP_WORDS
    ]

    return clean_tokens


# ============================================================
# 3) Préparer les produits
# ============================================================

def prepare_products(product_df):
    """
    On combine plusieurs colonnes du produit dans une seule colonne document_text.

    Colonnes utilisées :
    - product_name
    - product_class
    - category hierarchy
    - product_description
    """
    product_df = product_df.copy()

    text_columns = [
        "product_name",
        "product_class",
        "category hierarchy",
        "product_description",
    ]

    for col in text_columns:
        product_df[col] = product_df[col].fillna("").astype(str)

    product_df["document_text"] = product_df[text_columns].agg(" ".join, axis=1)

    return product_df


# ============================================================
# 4) Construire l'index inversé
# ============================================================

def build_index(product_df):
    """
    Index inversé :
    mot -> {product_id: fréquence_du_mot_dans_ce_produit}

    Exemple :
    "chair" -> {100: 2, 200: 1}
    """
    inverted_index = defaultdict(dict)
    document_lengths = {}

    for _, row in product_df.iterrows():
        product_id = row["product_id"]
        tokens = preprocess(row["document_text"])

        word_counts = Counter(tokens)
        document_lengths[product_id] = len(tokens)

        for word, frequency in word_counts.items():
            inverted_index[word][product_id] = frequency

    return inverted_index, document_lengths


# ============================================================
# 5) Fréquence des mots
# ============================================================

def get_word_frequency_table(inverted_index, top_n=30):
    """
    Affiche les mots les plus fréquents dans tous les produits.

    total_frequency = nombre total d'apparitions du mot
    document_frequency = nombre de produits contenant ce mot
    """
    rows = []

    for word, postings in inverted_index.items():
        rows.append(
            {
                "word": word,
                "total_frequency": sum(postings.values()),
                "document_frequency": len(postings),
            }
        )

    frequency_df = pd.DataFrame(rows)

    if frequency_df.empty:
        return frequency_df

    frequency_df = frequency_df.sort_values("total_frequency", ascending=False)
    return frequency_df.head(top_n)


def get_query_terms_frequency(query, inverted_index, product_id):
    """
    Affiche combien de fois les mots de la query apparaissent dans un produit.

    Exemple :
    query = "coffee table"
    résultat = "coffee:2 | table:1"
    """
    query_terms = preprocess(query)

    parts = []
    for term in query_terms:
        frequency = inverted_index.get(term, {}).get(product_id, 0)
        parts.append(f"{term}:{frequency}")

    return " | ".join(parts)


# ============================================================
# 6) IDF pour TF-IDF et BM25
# ============================================================

def prepare_idf(inverted_index, total_documents):
    """
    Calcule :
    - IDF pour TF-IDF
    - IDF pour BM25
    """
    idf_tfidf = {}
    idf_bm25 = {}

    for word, postings in inverted_index.items():
        df = len(postings)

        # IDF classique lissé pour TF-IDF
        idf_tfidf[word] = math.log((total_documents + 1) / (df + 1)) + 1

        # IDF spécifique BM25
        idf_bm25[word] = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))

    return idf_tfidf, idf_bm25


# ============================================================
# 7) Recherche TF-IDF
# ============================================================

def search_tfidf(query, inverted_index, document_lengths, idf_tfidf, top_n=10):
    """
    Recherche TF-IDF simple.

    Score TF-IDF =
    somme des TF-IDF des mots de la query dans le produit.

    TF = fréquence du mot / longueur du document
    IDF = importance du mot dans le corpus
    """
    query_terms = preprocess(query)
    scores = defaultdict(float)

    for term in query_terms:
        if term not in inverted_index:
            continue

        postings = inverted_index[term]

        for product_id, frequency in postings.items():
            document_length = document_lengths[product_id] or 1

            tf = frequency / document_length
            score = tf * idf_tfidf[term]

            scores[product_id] += score

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_results[:top_n]


# ============================================================
# 8) Recherche BM25
# ============================================================

def search_bm25(
    query,
    inverted_index,
    document_lengths,
    idf_bm25,
    average_document_length,
    top_n=10,
    k1=1.5,
    b=0.75,
):
    """
    Recherche BM25 simple.

    BM25 tient compte :
    - de la fréquence du mot
    - de l'IDF
    - de la longueur du document
    """
    query_counts = Counter(preprocess(query))
    scores = defaultdict(float)

    for term, query_frequency in query_counts.items():
        if term not in inverted_index:
            continue

        postings = inverted_index[term]

        for product_id, frequency in postings.items():
            document_length = document_lengths[product_id] or 1

            denominator = frequency + k1 * (
                1 - b + b * (document_length / average_document_length)
            )

            bm25_score = idf_bm25[term] * (
                (frequency * (k1 + 1)) / denominator
            )

            scores[product_id] += query_frequency * bm25_score

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_results[:top_n]


# ============================================================
# 9) Recherche Jaccard
# ============================================================

def jaccard_similarity(query, document_text):
    """
    Similarité de Jaccard classique.

    Jaccard = mots en commun / tous les mots différents

    Important :
    Jaccard utilise set(), donc il ne compte pas les répétitions.
    Exemple :
    "offre offre offre" devient {"offre"}.
    """
    query_words = set(preprocess(query))
    document_words = set(preprocess(document_text))

    if not query_words or not document_words:
        return 0.0

    intersection = query_words.intersection(document_words)
    union = query_words.union(document_words)

    score = len(intersection) / len(union)

    return score


def search_jaccard(query, product_df, top_n=10):
    """
    Recherche avec Jaccard.

    Pour chaque produit :
    - on compare les mots uniques de la query
    - avec les mots uniques du document produit
    """
    scores = []

    for _, row in product_df.iterrows():
        product_id = row["product_id"]
        document_text = row["document_text"]

        score = jaccard_similarity(query, document_text)

        scores.append((product_id, score))

    ranked_results = sorted(scores, key=lambda x: x[1], reverse=True)

    return ranked_results[:top_n]


# ============================================================
# 10) Comparer TF-IDF, BM25 et Jaccard
# ============================================================

def build_comparison_table(
    query_df_3,
    product_df,
    inverted_index,
    document_lengths,
    idf_tfidf,
    idf_bm25,
    top_n=10,
):
    """
    Compare TF-IDF, BM25 et Jaccard pour les 3 premières queries.
    """
    product_by_id = product_df.set_index("product_id", drop=False)
    average_document_length = sum(document_lengths.values()) / len(document_lengths)

    rows = []

    for _, query_row in query_df_3.iterrows():
        query_id = query_row["query_id"]
        query = query_row["query"]

        methods = {
            "TF-IDF": search_tfidf(
                query,
                inverted_index,
                document_lengths,
                idf_tfidf,
                top_n=top_n,
            ),
            "BM25": search_bm25(
                query,
                inverted_index,
                document_lengths,
                idf_bm25,
                average_document_length,
                top_n=top_n,
            ),
            "Jaccard": search_jaccard(
                query,
                product_df,
                top_n=top_n,
            ),
        }

        for method_name, results in methods.items():
            for rank, (product_id, score) in enumerate(results, start=1):
                product = product_by_id.loc[product_id]

                rows.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "method": method_name,
                        "rank": rank,
                        "product_id": product_id,
                        "score": round(float(score), 6),
                        "product_name": product["product_name"],
                        "product_class": product["product_class"],
                        "query_term_frequencies": get_query_terms_frequency(
                            query,
                            inverted_index,
                            product_id,
                        ),
                        "preview": str(product["product_description"])[:160],
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# 11) Lancer tout
# ============================================================

if __name__ == "__main__":
    product_df, query_df_3 = load_data()
    product_df = prepare_products(product_df)

    inverted_index, document_lengths = build_index(product_df)

    total_documents = len(product_df)
    idf_tfidf, idf_bm25 = prepare_idf(inverted_index, total_documents)

    print("Nombre de produits :", len(product_df))
    print("Les 3 queries utilisées :")
    print(query_df_3[["query_id", "query"]])

    print("\nMots les plus fréquents :")
    print(get_word_frequency_table(inverted_index, top_n=20))

    comparison_df = build_comparison_table(
        query_df_3,
        product_df,
        inverted_index,
        document_lengths,
        idf_tfidf,
        idf_bm25,
        top_n=10,
    )

    comparison_df.to_csv("resultats_tfidf_bm25_jaccard_top10.csv", index=False)

    print("\nRésultats TF-IDF vs BM25 vs Jaccard :")
    print(comparison_df)
    print("\nFichier créé : resultats_tfidf_bm25_jaccard_top10.csv")
