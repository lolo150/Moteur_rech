import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1) Configuration Streamlit
# ============================================================

st.set_page_config(
    page_title="Comparaison TF-IDF vs BM25 vs Jaccard",
    layout="wide"
)

st.title("Moteur de recherche : TF-IDF vs BM25 vs Jaccard")
st.write(
    "Cette application utilise `product.csv` et `query.csv` pour comparer "
    "les résultats de recherche avec TF-IDF, BM25 et Jaccard."
)


# ============================================================
# 2) Prétraitement du texte
# ============================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "with", "you", "your", "can", "will", "its", "have", "has", "had",
    "but", "not", "into", "more", "than", "such"
}


def preprocess(text):
    """
    Nettoie le texte :
    - minuscules
    - suppression ponctuation
    - suppression stop words
    """
    if pd.isna(text):
        return []

    text = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+", text)

    return [
        token
        for token in tokens
        if len(token) > 1 and token not in STOP_WORDS
    ]


# ============================================================
# 3) Chargement CSV
# ============================================================

def load_csv_file(uploaded_file, default_path):
    """
    Si l'utilisateur importe un fichier, on le lit.
    Sinon, on cherche un fichier local dans le dossier courant.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep="\t")

    path = Path(default_path)
    if path.exists():
        return pd.read_csv(path, sep="\t")

    # Cas où le fichier s'appelle product(1).csv ou query(1).csv
    if default_path == "product.csv" and Path("product(1).csv").exists():
        return pd.read_csv("product(1).csv", sep="\t")

    if default_path == "query.csv" and Path("query(1).csv").exists():
        return pd.read_csv("query(1).csv", sep="\t")

    return None


# ============================================================
# 4) Préparer les produits
# ============================================================

def prepare_products(product_df, selected_columns):
    """
    Combine les colonnes sélectionnées dans une seule colonne document_text.
    """
    product_df = product_df.copy()

    for col in selected_columns:
        product_df[col] = product_df[col].fillna("").astype(str)

    product_df["document_text"] = product_df[selected_columns].agg(" ".join, axis=1)

    return product_df


# ============================================================
# 5) Construire l'index inversé
# ============================================================

def build_index(product_df, product_id_col):
    """
    Index inversé :
    mot -> {product_id: fréquence_du_mot_dans_ce_produit}
    """
    inverted_index = defaultdict(dict)
    document_lengths = {}

    for _, row in product_df.iterrows():
        product_id = row[product_id_col]
        tokens = preprocess(row["document_text"])

        word_counts = Counter(tokens)
        document_lengths[product_id] = len(tokens)

        for word, frequency in word_counts.items():
            inverted_index[word][product_id] = frequency

    return inverted_index, document_lengths


# ============================================================
# 6) Fréquence des mots
# ============================================================

def get_word_frequency_table(inverted_index, top_n=30):
    rows = []

    for word, postings in inverted_index.items():
        rows.append(
            {
                "word": word,
                "total_frequency": sum(postings.values()),
                "document_frequency": len(postings),
            }
        )

    freq_df = pd.DataFrame(rows)

    if freq_df.empty:
        return freq_df

    return freq_df.sort_values("total_frequency", ascending=False).head(top_n)


def get_query_terms_frequency(query, product_id, inverted_index):
    terms = preprocess(query)

    parts = []
    for term in terms:
        frequency = inverted_index.get(term, {}).get(product_id, 0)
        parts.append(f"{term}:{frequency}")

    return " | ".join(parts)


# ============================================================
# 7) IDF pour TF-IDF et BM25
# ============================================================

def prepare_idf(inverted_index, total_documents):
    idf_tfidf = {}
    idf_bm25 = {}

    for word, postings in inverted_index.items():
        df = len(postings)

        idf_tfidf[word] = math.log((total_documents + 1) / (df + 1)) + 1
        idf_bm25[word] = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))

    return idf_tfidf, idf_bm25


# ============================================================
# 8) Recherche TF-IDF
# ============================================================

def search_tfidf(query, inverted_index, document_lengths, idf_tfidf, top_n=10):
    query_terms = preprocess(query)
    scores = defaultdict(float)

    for term in query_terms:
        if term not in inverted_index:
            continue

        postings = inverted_index[term]

        for product_id, frequency in postings.items():
            document_length = document_lengths[product_id] or 1

            tf = frequency / document_length
            tfidf_score = tf * idf_tfidf[term]

            scores[product_id] += tfidf_score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ============================================================
# 9) Recherche BM25
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

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ============================================================
# 10) Recherche Jaccard
# ============================================================

def jaccard_similarity(query, document_text):
    """
    Jaccard classique :
    mots en commun / tous les mots différents

    Attention :
    Jaccard ne compte pas les répétitions.
    """
    query_words = set(preprocess(query))
    document_words = set(preprocess(document_text))

    if not query_words or not document_words:
        return 0.0

    intersection = query_words.intersection(document_words)
    union = query_words.union(document_words)

    return len(intersection) / len(union)


def search_jaccard(query, product_df, product_id_col, top_n=10):
    scores = []

    for _, row in product_df.iterrows():
        product_id = row[product_id_col]
        document_text = row["document_text"]

        score = jaccard_similarity(query, document_text)

        scores.append((product_id, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]


# ============================================================
# 11) Tableau de résultats
# ============================================================

def build_results_table(
    query,
    method_name,
    ranked_results,
    product_df,
    product_id_col,
    product_name_col,
    product_class_col,
    description_col,
    inverted_index,
):
    product_by_id = product_df.set_index(product_id_col, drop=False)

    rows = []

    for rank, (product_id, score) in enumerate(ranked_results, start=1):
        product = product_by_id.loc[product_id]

        rows.append(
            {
                "rank": rank,
                "method": method_name,
                "product_id": product_id,
                "score": round(float(score), 6),
                "product_name": product[product_name_col],
                "product_class": product[product_class_col] if product_class_col else "",
                "query_term_frequencies": get_query_terms_frequency(
                    query,
                    product_id,
                    inverted_index,
                ),
                "preview": str(product[description_col])[:180] if description_col else "",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 12) Interface Streamlit
# ============================================================

st.sidebar.header("Fichiers")

uploaded_product = st.sidebar.file_uploader("Importer product.csv", type=["csv"])
uploaded_query = st.sidebar.file_uploader("Importer query.csv", type=["csv"])

product_df = load_csv_file(uploaded_product, "product.csv")
query_df = load_csv_file(uploaded_query, "query.csv")

if product_df is None or query_df is None:
    st.warning(
        "Ajoute `product.csv` et `query.csv` dans le même dossier que `streamlit_app_jaccard.py`, "
        "ou importe-les avec les boutons à gauche."
    )
    st.stop()


st.sidebar.header("Colonnes")

all_product_columns = list(product_df.columns)
all_query_columns = list(query_df.columns)

default_product_id = "product_id" if "product_id" in all_product_columns else all_product_columns[0]
default_product_name = "product_name" if "product_name" in all_product_columns else all_product_columns[0]
default_product_class = "product_class" if "product_class" in all_product_columns else all_product_columns[0]
default_description = "product_description" if "product_description" in all_product_columns else all_product_columns[0]
default_query_col = "query" if "query" in all_query_columns else all_query_columns[0]

product_id_col = st.sidebar.selectbox(
    "Colonne product_id",
    all_product_columns,
    index=all_product_columns.index(default_product_id),
)

product_name_col = st.sidebar.selectbox(
    "Colonne nom du produit",
    all_product_columns,
    index=all_product_columns.index(default_product_name),
)

product_class_col = st.sidebar.selectbox(
    "Colonne classe produit",
    all_product_columns,
    index=all_product_columns.index(default_product_class),
)

description_col = st.sidebar.selectbox(
    "Colonne description",
    all_product_columns,
    index=all_product_columns.index(default_description),
)

query_col = st.sidebar.selectbox(
    "Colonne query",
    all_query_columns,
    index=all_query_columns.index(default_query_col),
)

default_text_columns = [
    col
    for col in [
        "product_name",
        "product_class",
        "category hierarchy",
        "product_description",
    ]
    if col in all_product_columns
]

selected_text_columns = st.sidebar.multiselect(
    "Colonnes utilisées pour la recherche",
    all_product_columns,
    default=default_text_columns,
)

top_n = st.sidebar.slider("Top N résultats", min_value=5, max_value=30, value=10)
number_queries = st.sidebar.slider("Nombre de queries depuis query.csv", min_value=1, max_value=10, value=3)

k1 = st.sidebar.slider("BM25 k1", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
b = st.sidebar.slider("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

if not selected_text_columns:
    st.error("Choisis au moins une colonne texte pour faire la recherche.")
    st.stop()


# ============================================================
# 13) Préparation des données
# ============================================================

product_df = prepare_products(product_df, selected_text_columns)
query_df_selected = query_df.head(number_queries).copy()

inverted_index, document_lengths = build_index(product_df, product_id_col)

total_documents = len(product_df)
idf_tfidf, idf_bm25 = prepare_idf(inverted_index, total_documents)

average_document_length = (
    sum(document_lengths.values()) / len(document_lengths)
    if document_lengths
    else 1
)


# ============================================================
# 14) Affichage général
# ============================================================

st.subheader("Données chargées")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Nombre de produits", len(product_df))

with col2:
    st.metric("Nombre de queries utilisées", len(query_df_selected))

with col3:
    st.metric("Nombre de mots dans l'index", len(inverted_index))


with st.expander("Voir les premières queries"):
    st.dataframe(query_df_selected, use_container_width=True)

with st.expander("Voir les mots les plus fréquents dans les produits"):
    frequency_df = get_word_frequency_table(inverted_index, top_n=30)
    st.dataframe(frequency_df, use_container_width=True)

    if not frequency_df.empty:
        st.bar_chart(frequency_df.set_index("word")["total_frequency"])


# ============================================================
# 15) Recherche personnalisée
# ============================================================

st.subheader("Recherche personnalisée")

manual_query = st.text_input(
    "Écris ta propre requête",
    value=str(query_df_selected.iloc[0][query_col]) if len(query_df_selected) else "",
)

if manual_query:
    tfidf_results = search_tfidf(
        manual_query,
        inverted_index,
        document_lengths,
        idf_tfidf,
        top_n=top_n,
    )

    bm25_results = search_bm25(
        manual_query,
        inverted_index,
        document_lengths,
        idf_bm25,
        average_document_length,
        top_n=top_n,
        k1=k1,
        b=b,
    )

    jaccard_results = search_jaccard(
        manual_query,
        product_df,
        product_id_col,
        top_n=top_n,
    )

    tfidf_df = build_results_table(
        manual_query,
        "TF-IDF",
        tfidf_results,
        product_df,
        product_id_col,
        product_name_col,
        product_class_col,
        description_col,
        inverted_index,
    )

    bm25_df = build_results_table(
        manual_query,
        "BM25",
        bm25_results,
        product_df,
        product_id_col,
        product_name_col,
        product_class_col,
        description_col,
        inverted_index,
    )

    jaccard_df = build_results_table(
        manual_query,
        "Jaccard",
        jaccard_results,
        product_df,
        product_id_col,
        product_name_col,
        product_class_col,
        description_col,
        inverted_index,
    )

    left, middle, right = st.columns(3)

    with left:
        st.markdown("### TF-IDF")
        st.dataframe(tfidf_df, use_container_width=True)

    with middle:
        st.markdown("### BM25")
        st.dataframe(bm25_df, use_container_width=True)

    with right:
        st.markdown("### Jaccard")
        st.dataframe(jaccard_df, use_container_width=True)


# ============================================================
# 16) Comparaison automatique avec query.csv
# ============================================================

st.subheader("Comparaison automatique avec les queries de query.csv")

for _, query_row in query_df_selected.iterrows():
    query = str(query_row[query_col])

    st.markdown("---")
    st.markdown(f"## Query : `{query}`")

    tfidf_results = search_tfidf(
        query,
        inverted_index,
        document_lengths,
        idf_tfidf,
        top_n=top_n,
    )

    bm25_results = search_bm25(
        query,
        inverted_index,
        document_lengths,
        idf_bm25,
        average_document_length,
        top_n=top_n,
        k1=k1,
        b=b,
    )

    jaccard_results = search_jaccard(
        query,
        product_df,
        product_id_col,
        top_n=top_n,
    )

    tfidf_df = build_results_table(
        query,
        "TF-IDF",
        tfidf_results,
        product_df,
        product_id_col,
        product_name_col,
        product_class_col,
        description_col,
        inverted_index,
    )

    bm25_df = build_results_table(
        query,
        "BM25",
        bm25_results,
        product_df,
        product_id_col,
        product_name_col,
        product_class_col,
        description_col,
        inverted_index,
    )

    jaccard_df = build_results_table(
        query,
        "Jaccard",
        jaccard_results,
        product_df,
        product_id_col,
        product_name_col,
        product_class_col,
        description_col,
        inverted_index,
    )

    left, middle, right = st.columns(3)

    with left:
        st.markdown("### TF-IDF")
        st.dataframe(tfidf_df, use_container_width=True)

    with middle:
        st.markdown("### BM25")
        st.dataframe(bm25_df, use_container_width=True)

    with right:
        st.markdown("### Jaccard")
        st.dataframe(jaccard_df, use_container_width=True)
