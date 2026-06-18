import streamlit as st
import re
import indexer
from corpus import corpus, jugements_pertinence
from indexer import build_inverted_index
from bm25 import BM25
from n_gram_tokenizer import NGramTokenizer
from tfidf import tfidf
from evaluation import calcule_precision_at_k

# Configuration de la page
st.set_page_config(layout="wide", page_title="Moteur de Recherche IR - Démo")

# --- STYLE CSS PERSONNALISÉ ---
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

    st.title("🔬 Moteur de Recherche textuel Multi-modéle")
    query = st.text_input("Saisissez votre requête :", value="deep learning")

    def afficher_colonne_resultats(title, scores, q, column):
        with column:
            st.subheader(title)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
            ids = [idx for idx, _ in ranked]
            
            p5 = calcule_precision_at_k(ids, q, jugements_pertinence, k=5)
            st.metric(label="Précision@5", value=f"{p5*100:.1f}%")
            st.write("---")
            
            for rank, (idx, score) in enumerate(ranked):
                is_pertinent = jugements_pertinence.get(q, {}).get(idx, 0) == 1
                status_icon = "✅" if is_pertinent else "❌"
                
                st.markdown(f"""
                <div class="doc-card">
                    <b>Rang {rank+1} - Doc {idx}</b> {status_icon}<br>
                    <span class="score-tag">Score: {score:.4f}</span><br>
                    <small>{corpus[idx][:120]}...</small>
                </div>
                """, unsafe_allow_html=True)

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
                stats_std = build_inverted_index(corpus)
                engine_std = BM25(stats_std, k1=k1, b=b_param)
                engine_std.tokenizer = standard_tokenizer
                scores_std = engine_std.get_scores(query)
                afficher_colonne_resultats("🔹 BM25 Standard", scores_std, query, colonnes[col_index])
                col_index += 1
                
            if show_bm25_ngram:
                ngram_tokenizer = NGramTokenizer(n=n_val, mode=mode_val)
                indexer.tokenize = ngram_tokenizer
                stats_ngram = build_inverted_index(corpus)
                engine_ngram = BM25(stats_ngram, k1=k1, b=b_param)
                engine_ngram.tokenizer = ngram_tokenizer
                scores_ngram = engine_ngram.get_scores(query)
                afficher_colonne_resultats("🚀 BM25 + N-Gram", scores_ngram, query, colonnes[col_index])
                col_index += 1
                
            if show_tfidf:
                scores_tfidf = tfidf(corpus, query)
                afficher_colonne_resultats("📊 TF-IDF", scores_tfidf, query, colonnes[col_index])
    else:
        st.info("Entrez une requête pour démarrer la recherche.")

# ==========================================
# PAGE 2 : ANALYSE DU CORPUS (PAGE ENTIÈRE)
# ==========================================
elif page == "📦 Analyse du Corpus":
    st.title("📦 Analyse Globale du Corpus de Test")
    st.write("Ce corpus synthétique sert de *benchmark* pour isoler les forces, faiblesses et cas limites de chaque algorithme face aux variations de longueur, aux répétitions abusives, aux fautes de frappe et à la proximité des expressions.")
    
    st.markdown("---")
    
    c_doc0, c_doc1 = st.columns(2)
    with c_doc0:
        st.markdown("<p class='corpus-title'>📄 Doc 0 : Court et dense</p>", unsafe_allow_html=True)
        st.info(f'"{corpus[0]}"')
        st.markdown("**Objectif :** Référence idéale pour la requête *'deep learning'*. Sa densité en mots-clés doit lui assurer la première place.")
        
    with c_doc1:
        st.markdown("<p class='corpus-title'>📄 Doc 1 : Très long et bavard (Test de Longueur)</p>", unsafe_allow_html=True)
        st.info(f'"{corpus[1][:110]}..." [suivi de 400 mots génériques]')
        st.markdown("""
        **Comportement attendu (Requête : *deep learning*) :**
        * ❌ **TF-IDF normalisé (linéaire) :** Il détruit le document. La fréquence est divisée mathématiquement par la longueur totale, rendant le score presque nul.
        * ✅ **BM25 :** Grâce au paramètre **b**, la pénalité de longueur est non linéaire. Le document conserve un score honorable.
        """)

    st.markdown("---")
    
    c_doc2, c_doc3 = st.columns(2)
    with c_doc2:
        st.markdown("<p class='corpus-title'>📄 Doc 2 : Répétition abusive (Keyword Spamming)</p>", unsafe_allow_html=True)
        st.info(f'"{corpus[2]}"')
        st.markdown("""
        **Comportement attendu (Requête : *learning*) :**
        * ❌ **TF-IDF :** Piégé par la répétition, il donne un score linéaire proportionnel aux 5 occurrences et classe ce spam au rang 1.
        * ✅ **BM25 :** Grâce au paramètre **k1**, le score plafonne dès les premières occurrences, permettant au Doc 0 de repasser devant.
        """)
        
    with c_doc3:
        st.markdown("<p class='corpus-title'>📄 Doc 3 : Variante morphologique / Faute de frappe</p>", unsafe_allow_html=True)
        st.info(f'"{corpus[3]}"')
        st.markdown("""
        **Comportement attendu (Requête : *deeep*) :**
        * ❌ **TF-IDF / BM25 Standards :** Échouent complètement (Score de 0) car ils recherchent une correspondance exacte.
        * ✅ **BM25 + N-Gram (Caractères) :** Découpe en sous-chaînes (ex: *dee*, *eep*), match le document malgré la faute, et remonte le résultat.
        """)

    st.markdown("---")
    
    # NOUVEAU CAS DE TEST : EXPRESSION EXACTE
    st.markdown("<p class='corpus-title'>📄 Doc 4 : Mots dispersés (Test d'expression exacte)</p>", unsafe_allow_html=True)
    st.info(f'"{corpus[4]}"')
    st.markdown("""
    **Comportement attendu (Requête : *deep learning*) :**
    * ⚠️ **BM25 / TF-IDF Standards :** Vont attribuer un score élevé à ce document car il contient individuellement les mots *"deep"* et *"learning"*, sans réaliser qu'ils sont séparés et hors-contexte.
    * ✅ **BM25 + N-Gram (Word, N=2) :** Génère le token unique `"deep_learning"`. Comme le Doc 4 possède les mots séparés, le bigramme exact n'existe pas chez lui. Le Doc 0 (qui possède l'expression exacte) verra son score s'envoler loin devant le Doc 4, triant ainsi parfaitement l'expression exacte.
    """)