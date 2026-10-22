"""Construit le notebook de présentation du TP2 sans dépendance supplémentaire."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4


RACINE = Path(__file__).resolve().parent
SORTIE = RACINE / "TP2_Filtrage_Images.ipynb"


def markdown(texte: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": texte.strip().splitlines(keepends=True),
    }


def code(texte: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": texte.strip().splitlines(keepends=True),
    }


def construire() -> None:
    cellules = [
        markdown(
            """
# TP n°2 — Filtrage d’images

**Module :** Vision Artificielle — EMSI Casa  
**Étudiant :** Anwar Ezzirani — Groupe G4  
**Année universitaire :** 2026–2027

Ce notebook présente les réponses du TP : convolution, noyaux de filtrage, bruit poivre et sel, filtres moyenneur, gaussien et médian, puis comparaison quantitative par PSNR.
"""
        ),
        markdown(
            """
## Préparation

Le fichier `aero1.jpg` demandé n’étant pas fourni avec le sujet, la solution utilise automatiquement une image naturelle incluse avec Matplotlib. Il suffit de placer `aero1.jpg` dans le dossier pour l’utiliser sans modifier le code.
"""
        ),
        code(
            """
import matplotlib.pyplot as plt
import numpy as np

from TP2_Anwar_Ezzirani_G4 import (
    NOYAU_CONTRASTE, NOYAU_LISSAGE, NOYAU_REPOUSSAGE,
    ajouter_bruit_gaussien, ajouter_bruit_poivre_sel,
    charger_image, contraste, filtre_gaussien, filtre_median,
    filtre_moyenneur_blur, filtre_moyenneur_filter2d,
    histogramme, lissage, niveau_de_gris, psnr, repoussage,
)

image, source = charger_image()
gris = niveau_de_gris(image)
print(source, image.shape, image.dtype)
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.title("Image de test")
plt.show()
"""
        ),
        markdown(
            """
## 1. Matrice de convolution

La fonction `convolution(m, c)` est définie dans le script principal. Elle :

1. ajoute un *zero-padding* pour conserver la taille ;
2. déplace le noyau sur chaque pixel ;
3. traite séparément les trois composantes RGB ;
4. arrondit et sature le résultat dans `[0, 255]`.

Le type de sortie reste `np.uint8`.
"""
        ),
        markdown(
            """
## 2. Lissage, contraste et repoussage

- **Lissage** : moyenne des neuf pixels du voisinage.
- **Contraste/netteté** : renforce le pixel central et soustrait ses quatre voisins directs.
- **Repoussage** : combine des coefficients négatifs et positifs pour produire un effet de relief.
"""
        ),
        code(
            """
print("Noyau de lissage :\n", NOYAU_LISSAGE)
print("Noyau de contraste :\n", NOYAU_CONTRASTE)
print("Noyau de repoussage :\n", NOYAU_REPOUSSAGE)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, data, title in zip(
    axes.flat,
    [image, lissage(image), contraste(image), repoussage(image)],
    ["Originale", "Lissage", "Contraste", "Repoussage"],
):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""
        ),
        markdown("![Résultats des noyaux](resultats/01_noyaux_convolution.png)"),
        markdown(
            """
## 3. Bruit poivre et sel

Le nombre de pixels blancs et noirs est tiré entre 300 et 10 000. Les positions dépendent du générateur aléatoire : sans graine fixe, le bruit change à chaque exécution. Une graine `42` est utilisée dans le rapport afin de rendre les résultats reproductibles.

L’histogramme bruité présente deux pics supplémentaires aux niveaux 0 et 255.
"""
        ),
        code(
            """
rng = np.random.default_rng(42)
n_sel = int(rng.integers(300, 10_001))
n_poivre = int(rng.integers(300, 10_001))
bruitee = ajouter_bruit_poivre_sel(gris, n_sel, n_poivre, seed=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(gris, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Originale")
axes[1].imshow(bruitee, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Poivre et sel")
axes[2].plot(histogramme(bruitee))
axes[2].set_xlim(0, 255)
axes[2].set_title("Histogramme bruité")
for ax in axes[:2]: ax.axis("off")
plt.tight_layout()
plt.show()
"""
        ),
        markdown("![Bruit et histogramme](resultats/02_bruit_histogramme.png)"),
        markdown(
            """
## 4. Filtre moyenneur

Les noyaux sont `np.ones((n, n)) / n²` pour `n = 3, 5, 7`. Les deux méthodes demandées, `filter2D` et `blur`, donnent ici le même résultat. Une grande fenêtre supprime davantage de bruit mais détruit plus de détails.
"""
        ),
        code(
            """
for taille in (3, 5, 7):
    methode_1 = filtre_moyenneur_filter2d(bruitee, taille)
    methode_2 = filtre_moyenneur_blur(bruitee, taille)
    ecart = np.max(np.abs(methode_1.astype(int) - methode_2.astype(int)))
    print(f"{taille}×{taille} : PSNR={psnr(gris, methode_1):.2f} dB ; écart des méthodes={ecart}")
"""
        ),
        markdown("![Filtres moyenneurs](resultats/03_filtres_moyenneurs.png)"),
        markdown(
            """
## 5. Filtre gaussien

Le filtre gaussien accorde plus de poids aux pixels proches du centre. Pour ce bruit impulsionnel, une petite valeur de sigma conserve mieux les contours ; les valeurs `5` et `10` provoquent un flou important.
"""
        ),
        markdown("![Moyenneur contre gaussien](resultats/04_moyenneur_vs_gaussien.png)\n\n![Influence de sigma](resultats/05_gaussien_sigmas.png)"),
        markdown(
            """
## 6. Filtre médian

Le filtre médian est le mieux adapté au bruit poivre et sel : les valeurs extrêmes 0 et 255 influencent peu la médiane. Avec les données du rapport, le médian 3×3 atteint environ **30 dB**, contre environ **26 dB** pour les filtres moyenneur et gaussien.
"""
        ),
        markdown("![Comparaison sur bruit poivre et sel](resultats/06_filtre_median_poivre_sel.png)"),
        markdown(
            """
## 7. Filtre médian et bruit gaussien

Le filtre médian améliore aussi une image affectée par un bruit gaussien, mais il n’est pas le plus efficace dans ce cas. Le filtre gaussien correspond mieux à la nature statistique du bruit et obtient ici le meilleur PSNR.
"""
        ),
        markdown("![Comparaison sur bruit gaussien](resultats/07_filtre_median_bruit_gaussien.png)"),
        markdown(
            """
## Conclusion

La convolution permet d’implémenter de nombreux traitements à partir d’un simple noyau. Le choix du filtre dépend du bruit :

- le filtre **médian** est particulièrement performant pour le bruit poivre et sel ;
- le filtre **gaussien** est mieux adapté au bruit gaussien ;
- augmenter la taille du voisinage réduit le bruit, mais augmente aussi le flou.

Le PSNR confirme les observations visuelles et permet une comparaison objective avec l’image originale.
"""
        ),
    ]

    notebook = {
        "cells": cellules,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    SORTIE.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook créé : {SORTIE}")


if __name__ == "__main__":
    construire()
