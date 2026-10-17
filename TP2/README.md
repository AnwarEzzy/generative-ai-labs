# TP2 — Modèles génératifs avec PyTorch sur MNIST

Ce dossier contient la première partie du TP : chargement de MNIST, classification discriminative, autoencodeur simple et autoencodeur variationnel (VAE). Le livrable principal est [TP2_MNIST_modeles_generatifs.ipynb](TP2_MNIST_modeles_generatifs.ipynb).

## Ouvrir dans VS Code

1. Installer [uv](https://docs.astral.sh/uv/) et VS Code avec les extensions Python et Jupyter.
2. Ouvrir ce dossier `TP2` dans VS Code.
3. Dans le terminal, lancer `uv sync` (Python 3.11 et les dépendances sont décrits dans `.python-version`, `pyproject.toml` et `uv.lock`).
4. Ouvrir le notebook et sélectionner le noyau `.venv/Scripts/python.exe` sous Windows, ou `.venv/bin/python` sous Linux/macOS.
5. Exécuter les cellules dans l'ordre. Au premier lancement, `torchvision` télécharge MNIST dans `data/`.

Si l'on part de zéro, les commandes équivalentes sont `uv init --python 3.11`, `uv venv`, puis `uv add ipykernel torch torchvision torcheval matplotlib numpy nbclient`. L'activation est facultative avec `uv run` ; sous PowerShell : `.venv\Scripts\Activate.ps1`.

## Contenu

- **Classifieur** : réseau 784→128→10, entropie croisée, accuracy et matrice de confusion sur 10 000 images de test.
- **AE** : espace latent 2D, MSE par pixel, reconstructions, nuage latent et interpolation.
- **VAE** : rééchantillonnage, BCE + KL par image, reconstructions, nuage latent, échantillons nouveaux et grille de décodage 2D.

La séparation MNIST standard est conservée : 60 000 images pour l'entraînement, 10 000 pour le test. Le code fixe la graine aléatoire à 42 ; les résultats peuvent varier légèrement selon le matériel et les versions de PyTorch. Les données téléchargées, environnements et poids éventuels sont ignorés par Git.

## Résultats de l'exécution de référence

Exécution CPU avec PyTorch 2.14.1, sur le jeu de test complet :

| Modèle | Entraînement | Mesure sur le test |
| --- | ---: | --- |
| Classifieur | 3 époques | Accuracy 95,21 % ; CE 0,1614 |
| Autoencodeur | 5 époques | MSE 0,0486 par pixel |
| VAE | 10 époques | Perte 150,68 par image ; BCE 144,92 ; KL 5,76 |

La métrique VAE de test reconstruit à partir de la moyenne latente `μ` pour rester déterministe. Les figures et les sorties d'entraînement sont enregistrées dans le notebook. Ces trois métriques correspondent à des objectifs différents et ne se comparent pas directement.
