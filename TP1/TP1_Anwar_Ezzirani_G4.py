"""TP n°1 - Introduction au traitement d'images (Vision Artificielle).

Étudiant : Anwar Ezzirani — Groupe 4

Solution autonome des questions du sujet. Le programme utilise uniquement
NumPy et Matplotlib et produit les figures du compte rendu dans ``resultats``.

Exécution :
    python TP1_solution.py
    python TP1_solution.py --image autre_image.png --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def charger_image(chemin: str | Path) -> np.ndarray:
    """Charge une image RGB et garantit des composantes uint8 dans [0, 255]."""
    image = plt.imread(chemin)

    # Matplotlib lit généralement un PNG sous forme de float dans [0, 1],
    # alors qu'un JPEG est souvent lu directement en uint8.
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être en niveaux de gris, RGB ou RGBA.")
    return image


# Question 1
def symetrie(im: np.ndarray) -> np.ndarray:
    """Renvoie la symétrie de l'image par rapport à son axe vertical."""
    n, p, _ = im.shape
    im_sym = np.zeros_like(im)
    for i in range(n):
        for j in range(p):
            im_sym[i, j] = im[i, p - 1 - j]
    return im_sym


# Question 2
def negatif(im: np.ndarray) -> np.ndarray:
    """Renvoie le négatif RGB de l'image."""
    n, p, _ = im.shape
    im_neg = np.zeros_like(im)
    for i in range(n):
        for j in range(p):
            im_neg[i, j] = 255 - im[i, j]
    return im_neg


# Le sujet passe directement de la question 2 à la question 4.
# Question 4
def niveaudegris(im: np.ndarray) -> np.ndarray:
    """Convertit une image RGB en luminance uint8 selon la norme du sujet."""
    n, p, _ = im.shape
    im_lum = np.zeros((n, p), dtype=np.uint8)
    for i in range(n):
        for j in range(p):
            r, g, b = im[i, j].astype(float)
            im_lum[i, j] = np.uint8(np.rint(0.2126 * r + 0.7152 * g + 0.0722 * b))
    return im_lum


def _image_canal(im: np.ndarray, canal: int) -> np.ndarray:
    """Conserve un canal RGB et met les deux autres à zéro."""
    resultat = np.zeros_like(im)
    resultat[:, :, canal] = im[:, :, canal]
    return resultat


# Question 5
def image_rouge(im: np.ndarray) -> np.ndarray:
    return _image_canal(im, 0)


# Question 6
def image_verte(im: np.ndarray) -> np.ndarray:
    return _image_canal(im, 1)


# Question 7
def image_bleue(im: np.ndarray) -> np.ndarray:
    return _image_canal(im, 2)


# Question 8
def histo(im: np.ndarray) -> np.ndarray:
    """Compte les pixels de chaque niveau de gris (0 à 255)."""
    if im.ndim != 2:
        raise ValueError("histo attend une image en niveaux de gris (matrice 2D).")
    h = np.zeros(256, dtype=np.int64)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            h[int(im[i, j])] += 1
    return h


# Question 9
def egaliser_histogramme(h: np.ndarray, gray_image: np.ndarray) -> np.ndarray:
    """Construit la table de conversion à partir de l'histogramme cumulé.

    La formule suit exactement l'indication du sujet :
        he(k) = 255 * histogramme_cumule(k) / nombre_de_pixels
    """
    if len(h) != 256:
        raise ValueError("L'histogramme doit contenir exactement 256 valeurs.")
    if gray_image.ndim != 2:
        raise ValueError("gray_image doit être une matrice 2D.")

    hc = np.zeros(256, dtype=np.float64)
    hc[0] = h[0]
    for i in range(1, 256):
        hc[i] = hc[i - 1] + h[i]
    he = (hc / gray_image.size) * 255
    return he


# Question 10
def augmenter_contraste(gray_image: np.ndarray) -> np.ndarray:
    """Applique l'égalisation d'histogramme et renvoie une image uint8."""
    h = histo(gray_image)
    he = egaliser_histogramme(h, gray_image)
    table_conversion = np.rint(he).clip(0, 255).astype(np.uint8)

    new_image = gray_image.copy()
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            new_image[i, j] = table_conversion[gray_image[i, j]]
    return new_image


def enregistrer_resultats(image: np.ndarray, dossier: Path, afficher: bool = False) -> None:
    """Exécute toutes les questions et enregistre les résultats graphiques."""
    dossier.mkdir(parents=True, exist_ok=True)

    image_sym = symetrie(image)
    image_neg = negatif(image)
    image_grise = niveaudegris(image)
    rouge = image_rouge(image)
    vert = image_verte(image)
    bleu = image_bleue(image)
    contraste = augmenter_contraste(image_grise)

    transformations = [
        (image, "Image originale", None),
        (image_sym, "Q1 — Symétrie verticale", None),
        (image_neg, "Q2 — Négatif", None),
        (image_grise, "Q4 — Niveaux de gris", "gray"),
        (rouge, "Q5 — Canal rouge", None),
        (vert, "Q6 — Canal vert", None),
        (bleu, "Q7 — Canal bleu", None),
        (contraste, "Q10 — Contraste augmenté", "gray"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))
    for axe, (donnees, titre, cmap) in zip(axes.flat, transformations):
        axe.imshow(donnees, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        axe.set_title(titre)
        axe.axis("off")
    fig.suptitle("TP1 — Introduction au traitement d'images", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    fig.savefig(dossier / "comparaison_transformations.png", dpi=160, bbox_inches="tight")

    h_avant = histo(image_grise)
    h_apres = histo(contraste)
    fig_h, axes_h = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    axes_h[0].plot(np.arange(256), h_avant, color="black")
    axes_h[0].set_title("Q8 — Histogramme avant égalisation")
    axes_h[1].plot(np.arange(256), h_apres, color="tab:blue")
    axes_h[1].set_title("Histogramme après égalisation")
    for axe in axes_h:
        axe.set_xlabel("Niveau de gris")
        axe.set_ylabel("Nombre de pixels")
        axe.set_xlim(0, 255)
        axe.grid(alpha=0.2)
    fig_h.tight_layout()
    fig_h.savefig(dossier / "histogrammes_avant_apres.png", dpi=160, bbox_inches="tight")

    plt.imsave(dossier / "symetrie.png", image_sym)
    plt.imsave(dossier / "negatif.png", image_neg)
    plt.imsave(dossier / "niveaux_de_gris.png", image_grise, cmap="gray", vmin=0, vmax=255)
    plt.imsave(dossier / "canal_rouge.png", rouge)
    plt.imsave(dossier / "canal_vert.png", vert)
    plt.imsave(dossier / "canal_bleu.png", bleu)
    plt.imsave(dossier / "contraste_augmente.png", contraste, cmap="gray", vmin=0, vmax=255)

    print(f"Dimensions : {image.shape}")
    print(f"Type : {image.dtype}")
    print(f"Pixels de l'histogramme : {h_avant.sum()} (attendu : {image_grise.size})")
    print(f"Somme après égalisation : {h_apres.sum()} (attendu : {contraste.size})")
    print(f"Résultats enregistrés dans : {dossier.resolve()}")

    if afficher:
        plt.show()
    else:
        plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solution du TP1 de vision artificielle")
    parser.add_argument("--image", default="Lenna_(test_image).png", help="Image d'entrée")
    parser.add_argument("--sortie", default="resultats", help="Dossier de sortie")
    parser.add_argument("--show", action="store_true", help="Afficher également les figures")
    args = parser.parse_args()

    image = charger_image(args.image)
    enregistrer_resultats(image, Path(args.sortie), args.show)


if __name__ == "__main__":
    main()
