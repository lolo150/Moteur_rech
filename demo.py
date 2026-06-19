import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Importation des modules de recherche avancés
import indexer
from corpus import corpus, jugements_pertinence
from indexer import build_inverted_index
from bm25 import BM25
from n_gram_tokenizer import NGramTokenizer
from tfidf import tfidf
from evaluation import calcule_precision_at_k
from autocomplete import NGramAutocomplete

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
# 2) Fonctions Utilitaires & Graphiques
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

def afficher_comparaison_courbes(k1_val, query_text, docs_collection):
    """Génère le graphique de saturation basé sur la vraie TF max du corpus."""
    # Nettoyage simple pour isoler le premier mot-clé de la requête
    words = re.findall(r"\w+", query_text.lower())
    if not words:
        return
    target_word = words[0] # On zoome sur le premier terme significatif

    # Trouver la fréquence maximale de ce mot dans TOUT le corpus
    max_tf = 0
    for doc in docs_collection:
        tokens = re.findall(r"\w+", doc.lower())
        tf_dans_doc = tokens.count(target_word)
        if tf_dans_doc > max_tf:
            max_tf = tf_dans_doc

    # Sécurité : si le mot n'existe pas ou n'apparaît qu'une fois, on met une échelle minimale de 5
    limite_x = max(max_tf + 2, 5)

    st.write("---")
    st.markdown(f"### 📈 Analyse de la Saturation Réelle (Terme analysé : `{target_word}`)")
    st.write(f"L'axe X est adapté dynamiquement. Fréquence maximale trouvée dans le corpus : **{max_tf} occurrences**.")
    
    # L'abscisse va maintenant de 0 jusqu'à la limite réelle détectée
    tf_range = np.linspace(0, limite_x, 500)
    idf_simule = 2.5
    
    score_tfidf = tf_range * 0.15 * idf_simule
    score_bm25 = idf_simule * ((tf_range * (k1_val + 1)) / (tf_range + k1_val))
    
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(tf_range, score_tfidf, label="TF-IDF (Croissance linéaire)", color="red", linewidth=2)
    ax.plot(tf_range, score_bm25, label=f"BM25 (Saturation k1 = {k1_val:.1f})", color="blue", linewidth=2.5)
    
    # Ligne verticale pointillée pour marquer le document le plus dense
    if max_tf > 0:
        ax.axvline(x=max_tf, color="purple", linestyle="--", alpha=0.7, label=f"TF Max Réelle ({max_tf})")

    ax.set_xlim(0, limite_x)
    ax.set_ylim(0, max(max(score_bm25), max(score_tfidf)) * 1.1)
    ax.set_xlabel(f"Term Frequency du mot '{target_word}'")
    ax.set_ylabel("Score accordé")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    st.pyplot(fig)



def afficher_benchmark_performance(n_actuel, mode_actuel):
    """Génère le graphique comparatif de performance pour les N-Grams."""
    st.write("---")
    st.markdown("### 🏆 Benchmark d'Efficacité : Unigrammes vs N-Grams")
    st.info(f"Analyse de la précision théorique avec votre configuration actuelle : **N={n_actuel} ({mode_actuel})**")

    # Scénarios
    labels = ['Cas 1: Faute ("deeep")', 'Cas 2: Phrase ("deep learning")']
    
    # Données théoriques basées sur les capacités de l'algo
    std_scores = [0, 40]  # BM25 Standard échoue sur la faute, bruité sur la phrase
    
    # Calcul dynamique de la performance N-Gram selon les réglages
    ngram_char = [0, 0]
    ngram_word = [0, 0]
    
    if mode_actuel == 'char':
        ngram_char[0] = 100 if n_actuel >= 2 else 0 # Efficace sur typo si N >= 2
        ngram_char[1] = 40 # Reste bruité sur les phrases
    else:
        ngram_word[0] = 0 # Le mode word ne corrige pas les fautes de lettres
        ngram_word[1] = 100 if n_actuel >= 2 else 40 # Efficace sur phrase si N >= 2

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, std_scores, width, label='BM25 Standard', color='#f1f5f9', edgecolor='#94a3b8')
    ax.bar(x, ngram_char, width, label='BM25 + N-Gram (Char)', color='#3b82f6')
    ax.bar(x + width, ngram_word, width, label='BM25 + N-Gram (Word)', color='#10b981')

    ax.set_ylabel('Précision théorique (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Ajout des étiquettes de pourcentage sur les barres
    for i in range(len(labels)):
        ax.text(x[i]-width, std_scores[i]+2, f'{std_scores[i]}%', ha='center', fontweight='bold')
        ax.text(x[i], ngram_char[i]+2, f'{ngram_char[i]}%', ha='center', fontweight='bold')
        ax.text(x[i]+width, ngram_word[i]+2, f'{ngram_word[i]}%', ha='center', fontweight='bold')

    st.pyplot(fig)

# ==========================================
# PAGE 1 : ÉVALUATION & RECHERCHE
# ==========================================
if page == "🔍 Évaluation & Recherche":
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        show_tfidf = st.toggle("TF-IDF", value=True)
        
        show_bm25_std = st.toggle("BM25 Standard", value=True)
        with st.expander("🔧 Paramètres BM25 (k1, b)", expanded=False):
            k1 = st.slider("k1 (Saturation)", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
            b_param = st.slider("b (Normalisation)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
        
        show_bm25_ngram = st.toggle("BM25 + N-Gram", value=True)
        with st.expander("Paramètres N-Gram", expanded=False):
            n_val = st.slider("Valeur de N", 1, 4, 2)
            mode_val = st.radio("Mode", ["word", "char"])
            
        st.markdown("---")
        st.subheader("📂 Source de Données")
        data_source = st.radio("Collection à utiliser", ["Corpus Synthétique (Mémoire)", "Fichiers CSV (product.csv)"])
        
        product_df = None
        selected_text_cols = []
        if data_source == "Fichiers CSV (product.csv)":
            uploaded_product = st.file_uploader("Importer product.csv", type=["csv"])
            product_df = load_csv_file(uploaded_product, "product.csv")
            if product_df is not None:
                all_cols = list(product_df.columns)
                selected_text_cols = st.multiselect("Colonnes pour l'indexation", all_cols, default=all_cols[:2])

    st.title("🔬 Moteur de Recherche textuel Multi-modèle")
    
    # Choix du corpus actif
    active_corpus = corpus
    if data_source == "Fichiers CSV (product.csv)" and product_df is not None and selected_text_cols:
        product_df['combined'] = product_df[selected_text_cols].fillna("").astype(str).agg(" ".join, axis=1)        
        active_corpus = product_df['combined'].tolist()

    # Initialisation de l'autocomplétion
    @st.cache_resource
    def build_autocomplete_engine(dataset):
        ac = NGramAutocomplete(n=3)
        ac.fit(dataset)
        return ac

    autocomplete_engine = build_autocomplete_engine(active_corpus)

    if "search_query" not in st.session_state:
        st.session_state.search_query = "deep learning"

    # Composants de recherche et toggle d'autocomplétion
    input_query = st.text_input("Saisissez votre requête :", value=st.session_state.search_query)
    enable_autocomplete = st.toggle("🔮 Activer l'autocomplétion dynamique", value=True)

    query = input_query
    if show_bm25_ngram and enable_autocomplete and input_query:
        suggestions = autocomplete_engine.suggest(input_query, limit=5)
        
        if suggestions:
            st.write("💡 *Suggestions (cliquez pour rechercher) :*")
            cols_sug = st.columns(len(suggestions))
            for i, sug in enumerate(suggestions):
                with cols_sug[i]:
                    if st.button(sug, key=f"sug_{sug}", use_container_width=True):
                        st.session_state.search_query = sug
                        st.rerun()
            
            query = st.session_state.search_query

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

    # Exécution des requêtes
    if query:
        algos_actifs = sum([show_bm25_std, show_bm25_ngram, show_tfidf])
        
        if algos_actifs == 0:
            st.warning("⚠️ Veuillez activer au moins un algorithme dans la barre latérale.")
        else:
            scores_std = []
            scores_ngram = []
            scores_tfidf = []
            
            if show_bm25_std:
                standard_tokenizer = lambda text: re.findall(r"\w+", text.lower())
                indexer.tokenize = standard_tokenizer
                stats_std = build_inverted_index(active_corpus)
                engine_std = BM25(stats_std, k1=k1, b=b_param)
                engine_std.tokenizer = standard_tokenizer
                scores_std = engine_std.get_scores(query)
                
            if show_bm25_ngram:
                ngram_tokenizer = NGramTokenizer(n=n_val, mode=mode_val)
                indexer.tokenize = ngram_tokenizer
                stats_ngram = build_inverted_index(active_corpus)
                engine_ngram = BM25(stats_ngram, k1=k1, b=b_param)
                engine_ngram.tokenizer = ngram_tokenizer
                scores_ngram = engine_ngram.get_scores(query)
                
            if show_tfidf:
                scores_tfidf = tfidf(active_corpus, query)

            tous_les_scores = scores_std + scores_ngram + scores_tfidf
            
            # Gestion d'affichage si aucun score n'émerge
            if len(tous_les_scores) > 0 and all(score == 0 for score in tous_les_scores):
                st.info("ℹ️ Aucun document pertinent trouvé pour cette requête (tous les scores sont égaux à 0).")
            else:
                colonnes = st.columns(algos_actifs)
                col_index = 0
                
                if show_bm25_std:
                    afficher_colonne_resultats("BM25 Standard", scores_std, query, colonnes[col_index], active_corpus)
                    col_index += 1
                    
                if show_bm25_ngram:
                    afficher_colonne_resultats("BM25 + N-Gram", scores_ngram, query, colonnes[col_index], active_corpus)
                    col_index += 1
                    
                if show_tfidf:
                    afficher_colonne_resultats("TF-IDF", scores_tfidf, query, colonnes[col_index], active_corpus)

                # Affichage du graphique interactif de saturation
                if show_tfidf and show_bm25_std:
                    afficher_comparaison_courbes(k1, query, active_corpus)

                if show_bm25_ngram:
                # On passe les paramètres N et Mode configurés dans l'expander
                    afficher_benchmark_performance(n_val, mode_val)                    
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
    if len(corpus) > 4:
        st.info(f'"{corpus[4]}"')
    st.markdown("""
    **Comportement attendu :**
    * ⚠️ **Modèles Unigrammes :** Surévaluent la dispersion car les mots sont séparés.
    * ✅ **BM25 + N-Gram (Word, N=2) :** Génère un token d'expression unifié `"deep_learning"` absent ici.
    """)