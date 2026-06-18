import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import streamlit as st

# Importation de vos modules de recherche avancés
import indexer
from corpus import corpus, jugements_pertinence
from indexer import build_inverted_index
from bm25 import BM25
from n_gram_tokenizer import NGramTokenizer
from tfidf import tfidf
from evaluation import calcule_precision_at_k

# ============================================================
# 1) Configuration Streamlit & Styles
# ============================================================
st.set_page_config(
    page_title="Moteur de Recherche IR - Démo Multi-modèle",
    layout="wide"
)

st.markdown("""
    <style>
    .doc-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e6e9ef;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }
    .score-tag {
        color: #007BFF;
        font-weight: bold;
    }
    .corpus-title {
        color: #2E4053;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION DE LA BARRE LATÉRALE ---
with st.sidebar:
    st.header("📍 Navigation")
    page = st.radio("Aller à :", ["🔍 Évaluation & Recherche", "📦 Analyse du Corpus"])
    st.markdown("---")

# ============================================================
# 2) Fonctions Utilitaires de Chargement de Données (V1)
# ============================================================
def load_csv_file(uploaded_file, default_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep="\t")
    path = Path(default_path)
    if path.exists():
        return pd.read_csv(path, sep="\t")
    if default_path == "product.csv" and Path("product(1).csv").exists():
        return pd.read_csv("product(1).csv", sep="\t")
    if default_path == "query.csv" and Path("query(1).csv").exists():
        return pd.read_csv("query(1).csv", sep="\t")
    return None

# ==========================================
# PAGE 1 : ÉVALUATION & RECHERCHE
# ==========================================
if page == "🔍 Évaluation & Recherche":
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("🖥️ Algorithmes à activer")
        
        show_tfidf = st.toggle("📊 TF-IDF", value=True)
        st.markdown("---")
        
        show_bm25_std = st.toggle("🔹 BM25 Standard", value=True)
        with st.expander("🔧 Paramètres BM25 (k1, b)", expanded=False):
            k1 = st.number_input("k1 (Saturation)", value=1.5)
            b_param = st.number_input("b (Normalisation)", value=0.75)
        
        st.markdown("---")
        show_bm25_ngram = st.toggle("🚀 BM25 + N-Gram", value=True)
        with st.expander("📝 Paramètres N-Gram", expanded=False):
            n_val = st.slider("Valeur de N", 1, 4, 2)
            mode_val = st.radio("Mode", ["word", "char"])
            
        st.markdown("---")
        st.subheader("📂 Source de Données")
        data_source = st.radio("Collection à utiliser", ["Corpus Synthétique (Mémoire)", "Fichiers CSV (product.csv)"])
        
        if data_source == "Fichiers CSV (product.csv)":
            uploaded_product = st.file_uploader("Importer product.csv", type=["csv"])
            product_df = load_csv_file(uploaded_product, "product.csv")
            if product_df is not None:
                all_cols = list(product_df.columns)
                selected_text_cols = st.multiselect("Colonnes pour l'indexation", all_cols, default=all_cols[:2])

    st.title("🔬 Moteur de Recherche textuel Multi-modèle")
    query = st.text_input("Saisissez votre requête :", value="deep learning")

    def afficher_colonne_resultats(title, scores, q, column, docs_collection):
        with column:
            st.subheader(title)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
            ids = [idx for idx, _ in ranked]
            
            p5 = calcule_precision_at_k(ids, q, jugements_pertinence, k=5)
            st.metric(label="Précision@5 (Si dispo.)", value=f"{p5*100:.1f}%")
            st.write("---")
            
            for rank, (idx, score) in enumerate(ranked):
                is_pertinent = jugements_pertinence.get(q, {}).get(idx, 0) == 1
                status_icon = "✅" if is_pertinent else "❌"
                
                text_preview = docs_collection[idx] if idx < len(docs_collection) else "ID hors limites"
                st.markdown(f"""
                <div class="doc-card">
                    <b>Rang {rank+1} - Doc {idx}</b> {status_icon}<br>
                    <span class="score-tag">Score: {score:.4f}</span><br>
                    <small>{text_preview[:120]}...</small>
                </div>
                """, unsafe_allow_html=True)

    # Sélection dynamique du corpus de travail
    active_corpus = corpus
    if data_source == "Fichiers CSV (product.csv)" and product_df is not None and selected_text_cols:
        product_df['combined'] = product_df[selected_text_cols].fillna("").astype(str).agg(" ".join, axis=1)        
        active_corpus = product_df['combined'].tolist()

    if query:
        algos_actifs = sum([show_bm25_std, show_bm25_ngram, show_tfidf])
        
        if algos_actifs == 0:
            st.warning("⚠️ Veuillez activer au moins un algorithme dans la barre latérale.")
        else:
            colonnes = st.columns(algos_actifs)
            col_index = 0
            
            if show_bm25_std:
                standard_tokenizer = lambda text: re.findall(r"\w+", text.lower())
                indexer.tokenize = standard_tokenizer
                stats_std = build_inverted_index(active_corpus)
                engine_std = BM25(stats_std, k1=k1, b=b_param)
                engine_std.tokenizer = standard_tokenizer
                scores_std = engine_std.get_scores(query)
                afficher_colonne_resultats("🔹 BM25 Standard", scores_std, query, colonnes[col_index], active_corpus)
                col_index += 1
                
            if show_bm25_ngram:
                ngram_tokenizer = NGramTokenizer(n=n_val, mode=mode_val)
                indexer.tokenize = ngram_tokenizer
                stats_ngram = build_inverted_index(active_corpus)
                engine_ngram = BM25(stats_ngram, k1=k1, b=b_param)
                engine_ngram.tokenizer = ngram_tokenizer
                scores_ngram = engine_ngram.get_scores(query)
                afficher_colonne_resultats("🚀 BM25 + N-Gram", scores_ngram, query, colonnes[col_index], active_corpus)
                col_index += 1
                
            if show_tfidf:
                scores_tfidf = tfidf(active_corpus, query)
                afficher_colonne_resultats("📊 TF-IDF", scores_tfidf, query, colonnes[col_index], active_corpus)
    else:
        st.info("Entrez une requête pour démarrer la recherche.")

# ==========================================
# PAGE 2 : ANALYSE DU CORPUS (PAGE ENTIÈRE)
# ==========================================
elif page == "📦 Analyse du Corpus":
    st.title("📦 Analyse Globale du Corpus de Test")
    st.write("Ce corpus synthétique sert de *benchmark* pour isoler les forces, faiblesses et cas limites de chaque algorithme.")
    
    st.markdown("---")
    
    c_doc0, c_doc1 = st.columns(2)
    with c_doc0:
        st.markdown("<p class='corpus-title'>📄 Doc 0 : Court et dense</p>", unsafe_allow_html=True)
        st.info(f'"{corpus[0]}"')
        st.markdown("**Objectif :** Référence idéale pour la requête *'deep learning'*.")
        
    with c_doc1:
        st.markdown("<p class='corpus-title'>📄 Doc 1 : Très long et bavard (Test de Longueur)</p>", unsafe_allow_html=True)
        st.info(f'"{corpus[1][:110]}..." [suivi de 400 mots]')
        st.markdown("""
        **Comportement attendu :**
        * ❌ **TF-IDF :** Pénalité linéaire trop lourde. Score presque nul.
        * ✅ **BM25 :** Grâce à **b**, la courbe s'amortit et préserve un score juste.
        """)

    st.markdown("---")
    
    c_doc2, c_doc3 = st.columns(2)
    with c_doc2:
        st.markdown("<p class='corpus-title'>📄 Doc 2 : Répétition abusive (Keyword Spamming)</p>", unsafe_allow_html=True)
        st.info(f'"{corpus[2]}"')
        st.markdown("""
        **Comportement attendu :**
        * ❌ **TF-IDF :** Piégé par la répétition (croissance purement linéaire).
        * ✅ **BM25 :** Grâce à **k1**, la fréquence sature rapidement.
        """)
        
    with c_doc3:
        st.markdown("<p class='corpus-title'>📄 Doc 3 : Variante morphologique / Faute de frappe</p>", unsafe_allow_html=True)
        st.info(f'"{corpus[3]}"')
        st.markdown("""
        **Comportement attendu :**
        * ❌ **Standards :** Donnent un score de 0 (pas de match exact).
        * ✅ **BM25 + N-Gram :** Capte les sous-chaînes partielles.
        """)

    st.markdown("---")
    st.markdown("<p class='corpus-title'>📄 Doc 4 : Mots dispersés (Test d'expression exacte)</p>", unsafe_allow_html=True)
    st.info(f'"{corpus[4]}"')
    st.markdown("""
    **Comportement attendu :**
    * ⚠️ **Modèles Unigrammes :** Surévaluent la dispersion car les mots sont séparés.
    * ✅ **BM25 + N-Gram (Word, N=2) :** Génère un token d'expression unifié `"deep_learning"` absent ici.
    """)