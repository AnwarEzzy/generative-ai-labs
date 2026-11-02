# TP3 — RAG multimodal agentique

Ce TP interroge le PDF *Attention Is All You Need* avec une recherche texte/image dans ChromaDB et un agent ReAct LangGraph utilisant un modèle local Ollama.

## Lancer le notebook

1. Installer Python 3.11 ou plus récent et [Ollama](https://ollama.com/).
2. Dans ce dossier, installer les dépendances : `python -m pip install -r requirements.txt`.
3. Télécharger le modèle local : `ollama pull qwen3-vl:4b`. Garder le serveur Ollama actif.
4. Ouvrir `TP3_RAG_multimodal_agentique.ipynb` dans VS Code, sélectionner le noyau Python contenant les dépendances et exécuter les cellules dans l'ordre.

Le notebook tente de télécharger le [PDF officiel](https://arxiv.org/pdf/1706.03762) au premier lancement. Si cela échoue, placer le document dans `pdfs/attention_is_all_you_need.pdf` puis relancer la cellule. Les poids CLIP sont également téléchargés au premier lancement.

Les textes extraits, images, index Chroma et PDF téléchargé restent dans `artifacts/` ou `pdfs/`, exclus de Git. Le notebook source fourni par le professeur est conservé localement comme référence et exclu du livrable.

## Correspondance avec le sujet

| Étape | Réalisation |
|---|---|
| 1 | Extraction du texte par page, des images intégrées et des rendus des pages |
| 2 | Embeddings texte/image CLIP `ViT-B-32`, normalisés |
| 3 | ChromaDB persistant, distance cosinus, identifiant SHA-256 et métadonnées |
| 4 | Requêtes texte, image et hybrides avec classement et aperçus |
| 5 | `retriever_tool` documenté pour l'agent |
| 6 | Graphe ReAct explicite MIND/BODY avec Qwen3-VL local et LangGraph |
| 7 | Tests de recherche et traces des appels d'outil de l'agent |

La génération complète exige le PDF, les poids CLIP et le modèle Ollama installés localement. Les réponses de l'agent doivent être confrontées aux pages citées.
