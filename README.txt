# Version complète TF-IDF + BM25 + Jaccard

## Fichiers inclus

- `simple_search_engine_jaccard.py` : code Python complet.
- `streamlit_app_jaccard.py` : application Streamlit complète.
- `comparaison_tfidf_bm25_jaccard_top10.ipynb` : notebook complet.
- `resultats_tfidf_bm25_jaccard_top10.csv` : résultats générés sur les 3 premières queries.
- `requirements.txt` : dépendances.

## Lancer le script Python

Mets `product.csv` et `query.csv` dans le même dossier, puis lance :

```bash
python simple_search_engine_jaccard.py
```

## Lancer Streamlit

```bash
pip install -r requirements.txt
streamlit run streamlit_app_jaccard.py
```

## Ce qui a été ajouté

La méthode `Jaccard` a été ajoutée à côté de `TF-IDF` et `BM25`.

Jaccard calcule :

```text
mots en commun / tous les mots différents
```

Important : Jaccard classique ne compte pas les répétitions.
