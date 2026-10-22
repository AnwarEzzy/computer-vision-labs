"""TP n°2 — Filtrage d'images.

Auteur : Anwar Ezzirani — Groupe G4

Le script répond à toutes les questions du sujet et enregistre les figures dans
``resultats``. Il utilise OpenCV lorsqu'il est installé, avec une implémentation
SciPy équivalente comme solution de secours.

Exécution :
    python TP2_Anwar_Ezzirani_G4.py
    python TP2_Anwar_Ezzirani_G4.py --image aero1.jpg --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cbook import get_sample_data
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter

try:
    import cv2  # type: ignore
except ImportError:  # OpenCV est optionnel : SciPy fournit les mêmes filtres.
    cv2 = None


RACINE = Path(__file__).resolve().parent


def charger_image(chemin: str | Path | None = None) -> tuple[np.ndarray, str]:
    """Charge l'image demandée, aero1.jpg, ou une image de démonstration.

    Le redimensionnement limite le temps de la convolution pédagogique, qui
    traite volontairement chaque pixel à l'aide de boucles.
    """
    candidats: list[Path] = []
    if chemin:
        candidats.append(Path(chemin))
    candidats.extend([RACINE / "aero1.jpg", RACINE / "aero1.png"])

    source: Path | None = next((p for p in candidats if p.exists()), None)
    if source is None:
        source = Path(get_sample_data("grace_hopper.jpg", asfileobj=False))
        description = "grace_hopper.jpg (image de démonstration Matplotlib)"
    else:
        description = source.name

    with Image.open(source) as image_pil:
        image_pil = image_pil.convert("RGB")
        image_pil.thumbnail((512, 512), Image.Resampling.LANCZOS)
        image = np.asarray(image_pil, dtype=np.uint8)
    return image, description


def niveau_de_gris(image: np.ndarray) -> np.ndarray:
    """Convertit une image RGB en luminance selon les coefficients du TP1."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=True)
    rgb = image[..., :3].astype(np.float64)
    gris = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.rint(gris).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Question 1 — Convolution
# ---------------------------------------------------------------------------
def convolution(m: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Calcule M ⊗ C avec zero-padding et saturation dans [0, 255].

    Le noyau peut avoir toute taille impaire. Les images en niveaux de gris et
    les images couleur sont acceptées ; chaque canal est traité séparément.
    """
    noyau = np.asarray(c, dtype=np.float64)
    if noyau.ndim != 2 or noyau.shape[0] % 2 == 0 or noyau.shape[1] % 2 == 0:
        raise ValueError("Le noyau doit être une matrice 2D de dimensions impaires.")
    if m.ndim not in (2, 3):
        raise ValueError("L'image doit être une matrice 2D ou 3D.")

    image = np.asarray(m)
    etait_grise = image.ndim == 2
    if etait_grise:
        image = image[..., np.newaxis]

    hauteur, largeur, canaux = image.shape
    rayon_y, rayon_x = noyau.shape[0] // 2, noyau.shape[1] // 2
    image_pad = np.pad(
        image.astype(np.float64),
        ((rayon_y, rayon_y), (rayon_x, rayon_x), (0, 0)),
        mode="constant",
    )
    resultat = np.zeros((hauteur, largeur, canaux), dtype=np.uint8)

    # Une corrélation est utilisée comme dans le pseudo-code du sujet. Pour les
    # noyaux symétriques du TP, convolution et corrélation donnent le même résultat.
    for i in range(hauteur):
        for j in range(largeur):
            voisinage = image_pad[i : i + noyau.shape[0], j : j + noyau.shape[1]]
            somme = np.sum(voisinage * noyau[..., np.newaxis], axis=(0, 1))
            resultat[i, j] = np.rint(somme).clip(0, 255).astype(np.uint8)

    return resultat[..., 0] if etait_grise else resultat


# ---------------------------------------------------------------------------
# Question 2 — Noyaux classiques
# ---------------------------------------------------------------------------
NOYAU_LISSAGE = np.ones((3, 3), dtype=np.float64) / 9.0
NOYAU_CONTRASTE = np.array(
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64
)
NOYAU_REPOUSSAGE = np.array(
    [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float64
)


def lissage(im: np.ndarray) -> np.ndarray:
    return convolution(im, NOYAU_LISSAGE)


def contraste(im: np.ndarray) -> np.ndarray:
    return convolution(im, NOYAU_CONTRASTE)


def repoussage(im: np.ndarray) -> np.ndarray:
    return convolution(im, NOYAU_REPOUSSAGE)


# ---------------------------------------------------------------------------
# Partie 2 — Bruit poivre et sel
# ---------------------------------------------------------------------------
def ajouter_bruit_poivre_sel(
    image: np.ndarray,
    nombre_sel: int,
    nombre_poivre: int,
    seed: int | None = None,
) -> np.ndarray:
    """Ajoute des pixels blancs et noirs à des positions aléatoires distinctes."""
    if image.ndim != 2:
        raise ValueError("Le bruit est appliqué ici à une image en niveaux de gris.")
    if nombre_sel < 0 or nombre_poivre < 0:
        raise ValueError("Les nombres de pixels bruités doivent être positifs.")

    total = image.size
    quantite = min(nombre_sel + nombre_poivre, total)
    nombre_sel = min(nombre_sel, quantite)
    nombre_poivre = quantite - nombre_sel
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=quantite, replace=False)

    resultat = image.copy().reshape(-1)
    resultat[indices[:nombre_sel]] = 255
    resultat[indices[nombre_sel:]] = 0
    return resultat.reshape(image.shape)


def ajouter_bruit_gaussien(
    image: np.ndarray, sigma: float = 25.0, seed: int | None = None
) -> np.ndarray:
    """Ajoute un bruit gaussien de moyenne nulle."""
    rng = np.random.default_rng(seed)
    bruit = rng.normal(0.0, sigma, size=image.shape)
    return np.rint(image.astype(np.float64) + bruit).clip(0, 255).astype(np.uint8)


def histogramme(image: np.ndarray) -> np.ndarray:
    """Renvoie l'histogramme entier à 256 niveaux."""
    return np.bincount(image.ravel(), minlength=256)


# ---------------------------------------------------------------------------
# Parties 3, 4 et 5 — Filtres moyenneur, gaussien et médian
# ---------------------------------------------------------------------------
def _taille_impaire(taille: int) -> None:
    if taille < 1 or taille % 2 == 0:
        raise ValueError("La taille du filtre doit être un entier impair positif.")


def filtre_moyenneur_filter2d(image: np.ndarray, taille: int) -> np.ndarray:
    """Méthode 1 du sujet : noyau explicite et filter2D."""
    _taille_impaire(taille)
    noyau = np.ones((taille, taille), dtype=np.float32) / (taille * taille)
    if cv2 is not None:
        return cv2.filter2D(image, -1, noyau, borderType=cv2.BORDER_REFLECT)
    # Solution de secours mathématiquement équivalente.
    resultat = uniform_filter(image.astype(np.float64), size=taille, mode="reflect")
    return np.rint(resultat).clip(0, 255).astype(np.uint8)


def filtre_moyenneur_blur(image: np.ndarray, taille: int) -> np.ndarray:
    """Méthode 2 du sujet : cv2.blur ou son équivalent SciPy."""
    _taille_impaire(taille)
    if cv2 is not None:
        return cv2.blur(image, (taille, taille), borderType=cv2.BORDER_REFLECT)
    resultat = uniform_filter(image.astype(np.float64), size=taille, mode="reflect")
    return np.rint(resultat).clip(0, 255).astype(np.uint8)


def filtre_gaussien(
    image: np.ndarray, taille: int | None = None, sigma: float = 0.0
) -> np.ndarray:
    """Applique GaussianBlur ; taille=None laisse le support dépendre de sigma."""
    if taille is not None:
        _taille_impaire(taille)
    if cv2 is not None:
        ksize = (0, 0) if taille is None else (taille, taille)
        return cv2.GaussianBlur(
            image, ksize, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
        )

    if sigma <= 0:
        if taille is None:
            raise ValueError("Il faut fournir taille ou sigma.")
        # Approximation de la valeur automatiquement choisie par OpenCV.
        sigma = 0.3 * ((taille - 1) * 0.5 - 1) + 0.8
    truncate = 4.0 if taille is None else ((taille - 1) / 2) / sigma
    resultat = gaussian_filter(
        image.astype(np.float64), sigma=sigma, mode="reflect", truncate=truncate
    )
    return np.rint(resultat).clip(0, 255).astype(np.uint8)


def filtre_median(image: np.ndarray, taille: int) -> np.ndarray:
    """Applique le filtre médian avec une fenêtre carrée impaire."""
    _taille_impaire(taille)
    if cv2 is not None:
        return cv2.medianBlur(image, taille)
    return median_filter(image, size=taille, mode="reflect").astype(np.uint8)


def psnr(reference: np.ndarray, image: np.ndarray) -> float:
    """Calcule le Peak Signal-to-Noise Ratio en décibels."""
    if reference.shape != image.shape:
        raise ValueError("Les deux images doivent avoir la même forme.")
    erreur = np.mean((reference.astype(np.float64) - image.astype(np.float64)) ** 2)
    return math.inf if erreur == 0 else 10 * math.log10((255.0**2) / erreur)


def _afficher(axe, image, titre: str, gris: bool = False) -> None:
    axe.imshow(image, cmap="gray" if gris else None, vmin=0 if gris else None, vmax=255 if gris else None)
    axe.set_title(titre, fontsize=10)
    axe.axis("off")


def executer_tp(
    chemin_image: str | Path | None = None,
    dossier_sortie: str | Path = RACINE / "resultats",
    seed: int = 42,
) -> dict[str, float]:
    """Exécute toutes les expériences, génère les figures et renvoie les PSNR."""
    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    image, source = charger_image(chemin_image)
    gris = niveau_de_gris(image)
    Image.fromarray(image).save(sortie / "image_test_utilisee.png")

    # Q1-Q2 : convolution et trois effets.
    image_lissee = lissage(image)
    image_contraste = contraste(image)
    image_repoussage = repoussage(image)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axe, donnees, titre in zip(
        axes.flat,
        [image, image_lissee, image_contraste, image_repoussage],
        ["Image originale", "Lissage 3×3", "Augmentation du contraste", "Repoussage"],
    ):
        _afficher(axe, donnees, titre)
    fig.suptitle("Questions 1–2 — Convolution et noyaux", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(sortie / "01_noyaux_convolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Partie 2 : le seed rend le rapport reproductible. Changer le seed change le bruit.
    rng = np.random.default_rng(seed)
    n_sel = int(rng.integers(300, 10_001))
    n_poivre = int(rng.integers(300, 10_001))
    bruitee = ajouter_bruit_poivre_sel(gris, n_sel, n_poivre, seed=seed)
    autre_bruit = ajouter_bruit_poivre_sel(gris, n_sel, n_poivre, seed=seed + 1)
    pixels_differents = int(np.count_nonzero(bruitee != autre_bruit))
    Image.fromarray(bruitee).save(sortie / "noisy_image.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _afficher(axes[0, 0], gris, "Image originale en niveaux de gris", True)
    _afficher(
        axes[0, 1],
        bruitee,
        f"Bruit poivre et sel\nsel={n_sel}, poivre={n_poivre}",
        True,
    )
    axes[1, 0].plot(np.arange(256), histogramme(gris), color="black", linewidth=1)
    axes[1, 0].set_title("Histogramme original")
    axes[1, 1].plot(np.arange(256), histogramme(bruitee), color="tab:red", linewidth=1)
    axes[1, 1].set_title("Histogramme bruité : pics en 0 et 255")
    for axe in axes[1]:
        axe.set_xlim(0, 255)
        axe.set_xlabel("Niveau de gris")
        axe.set_ylabel("Nombre de pixels")
        axe.grid(alpha=0.2)
    fig.suptitle("Partie 2 — Bruit poivre et sel", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(sortie / "02_bruit_histogramme.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Partie 3 : filtres moyenneurs 3×3, 5×5 et 7×7.
    tailles = (3, 5, 7)
    moyenneurs = {t: filtre_moyenneur_filter2d(bruitee, t) for t in tailles}
    moyenneurs_blur = {t: filtre_moyenneur_blur(bruitee, t) for t in tailles}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    _afficher(axes[0], bruitee, "Image bruitée", True)
    for axe, taille in zip(axes[1:], tailles):
        _afficher(axe, moyenneurs[taille], f"Moyenneur {taille}×{taille}\nPSNR={psnr(gris, moyenneurs[taille]):.2f} dB", True)
    fig.suptitle("Partie 3 — Influence de la taille du filtre moyenneur", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(sortie / "03_filtres_moyenneurs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Partie 4 : moyenneur et gaussien, à tailles correspondantes.
    gaussiens_taille = {t: filtre_gaussien(bruitee, taille=t, sigma=0) for t in tailles}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, taille in enumerate(tailles):
        _afficher(axes[0, col], moyenneurs[taille], f"Moyenneur {taille}×{taille}\n{psnr(gris, moyenneurs[taille]):.2f} dB", True)
        _afficher(axes[1, col], gaussiens_taille[taille], f"Gaussien {taille}×{taille}\n{psnr(gris, gaussiens_taille[taille]):.2f} dB", True)
    fig.suptitle("Partie 4 — Comparaison moyenneur / gaussien", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(sortie / "04_moyenneur_vs_gaussien.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sigmas = (1.0, 5.0, 10.0)
    gaussiens_sigma = {s: filtre_gaussien(bruitee, taille=None, sigma=s) for s in sigmas}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    _afficher(axes[0], bruitee, "Image bruitée", True)
    for axe, sigma in zip(axes[1:], sigmas):
        resultat = gaussiens_sigma[sigma]
        _afficher(axe, resultat, f"Gaussien σ={sigma:g}\nPSNR={psnr(gris, resultat):.2f} dB", True)
    fig.suptitle("Partie 4 — Influence de l’écart-type σ", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(sortie / "05_gaussien_sigmas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Partie 5 : filtre médian contre le bruit poivre et sel.
    medians = {t: filtre_median(bruitee, t) for t in tailles}
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    donnees = [
        (bruitee, "Image bruitée"),
        (medians[3], f"Médian 3×3\n{psnr(gris, medians[3]):.2f} dB"),
        (medians[5], f"Médian 5×5\n{psnr(gris, medians[5]):.2f} dB"),
        (medians[7], f"Médian 7×7\n{psnr(gris, medians[7]):.2f} dB"),
        (moyenneurs[5], f"Moyenneur 5×5\n{psnr(gris, moyenneurs[5]):.2f} dB"),
        (gaussiens_taille[5], f"Gaussien 5×5\n{psnr(gris, gaussiens_taille[5]):.2f} dB"),
    ]
    for axe, (donnees_image, titre) in zip(axes.flat, donnees):
        _afficher(axe, donnees_image, titre, True)
    fig.suptitle("Partie 5 — Filtres appliqués au bruit poivre et sel", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(sortie / "06_filtre_median_poivre_sel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Test demandé : le filtre médian est-il aussi adapté au bruit gaussien ?
    bruit_gaussien = ajouter_bruit_gaussien(gris, sigma=25, seed=seed)
    median_gauss = filtre_median(bruit_gaussien, 5)
    moyen_gauss = filtre_moyenneur_blur(bruit_gaussien, 5)
    filtre_gauss = filtre_gaussien(bruit_gaussien, taille=5, sigma=0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    comparaisons = [
        (bruit_gaussien, f"Bruit gaussien\n{psnr(gris, bruit_gaussien):.2f} dB"),
        (median_gauss, f"Médian 5×5\n{psnr(gris, median_gauss):.2f} dB"),
        (moyen_gauss, f"Moyenneur 5×5\n{psnr(gris, moyen_gauss):.2f} dB"),
        (filtre_gauss, f"Gaussien 5×5\n{psnr(gris, filtre_gauss):.2f} dB"),
    ]
    for axe, (donnees_image, titre) in zip(axes, comparaisons):
        _afficher(axe, donnees_image, titre, True)
    fig.suptitle("Partie 5 — Test sur un bruit gaussien", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(sortie / "07_filtre_median_bruit_gaussien.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    mesures: dict[str, float] = {"Bruit poivre et sel": psnr(gris, bruitee)}
    mesures.update({f"Moyenneur {t}x{t}": psnr(gris, moyenneurs[t]) for t in tailles})
    mesures.update({f"Gaussien {t}x{t}": psnr(gris, gaussiens_taille[t]) for t in tailles})
    mesures.update({f"Gaussien sigma={s:g}": psnr(gris, gaussiens_sigma[s]) for s in sigmas})
    mesures.update({f"Median {t}x{t}": psnr(gris, medians[t]) for t in tailles})
    mesures.update(
        {
            "Bruit gaussien sigma=25": psnr(gris, bruit_gaussien),
            "Median 5x5 sur bruit gaussien": psnr(gris, median_gauss),
            "Moyenneur 5x5 sur bruit gaussien": psnr(gris, moyen_gauss),
            "Gaussien 5x5 sur bruit gaussien": psnr(gris, filtre_gauss),
        }
    )

    with (sortie / "mesures_psnr.csv").open("w", newline="", encoding="utf-8-sig") as fichier:
        writer = csv.writer(fichier, delimiter=";")
        writer.writerow(["Méthode", "PSNR (dB)"])
        for nom, valeur in mesures.items():
            writer.writerow([nom, f"{valeur:.4f}"])

    ecarts_methodes = {
        t: int(np.max(np.abs(moyenneurs[t].astype(int) - moyenneurs_blur[t].astype(int))))
        for t in tailles
    }
    backend = "OpenCV" if cv2 is not None else "SciPy (équivalent OpenCV)"
    print(f"Image utilisée : {source}")
    print(f"Dimensions : {image.shape} — type : {image.dtype}")
    print(f"Backend des filtres : {backend}")
    print(f"Bruit : {n_sel} pixels sel et {n_poivre} pixels poivre")
    print(f"Deux exécutions avec des graines différentes : {pixels_differents} pixels différents")
    print(f"Écart maximal filter2D / blur : {ecarts_methodes}")
    print("PSNR sur le bruit poivre et sel :")
    for nom, valeur in mesures.items():
        if "bruit gaussien" not in nom.lower():
            print(f"  {nom:<28} {valeur:6.2f} dB")
    print(f"Résultats enregistrés dans : {sortie.resolve()}")
    return mesures


def main() -> None:
    parser = argparse.ArgumentParser(description="Solution complète du TP2 de vision artificielle")
    parser.add_argument("--image", help="Chemin de l'image ; aero1.jpg est recherché par défaut")
    parser.add_argument("--sortie", default=str(RACINE / "resultats"), help="Dossier de sortie")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire du rapport")
    args = parser.parse_args()
    executer_tp(args.image, args.sortie, args.seed)


if __name__ == "__main__":
    main()
