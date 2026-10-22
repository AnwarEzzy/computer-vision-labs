"""Tests fonctionnels légers pour les fonctions du TP2."""

import numpy as np

from TP2_Anwar_Ezzirani_G4 import (
    ajouter_bruit_poivre_sel,
    convolution,
    filtre_gaussien,
    filtre_median,
    filtre_moyenneur_blur,
    histogramme,
    niveau_de_gris,
    psnr,
)


def test_convolution_identite() -> None:
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    noyau = np.zeros((3, 3), dtype=float)
    noyau[1, 1] = 1
    assert np.array_equal(convolution(image, noyau), image)


def test_saturation_convolution() -> None:
    image = np.full((3, 3), 200, dtype=np.uint8)
    resultat = convolution(image, np.ones((3, 3)))
    assert resultat.dtype == np.uint8
    assert resultat[1, 1] == 255


def test_niveau_de_gris_et_histogramme() -> None:
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    gris = niveau_de_gris(image)
    assert gris.shape == (1, 3)
    assert histogramme(gris).sum() == 3


def test_bruit_reproductible() -> None:
    image = np.full((20, 20), 127, dtype=np.uint8)
    a = ajouter_bruit_poivre_sel(image, 20, 30, seed=7)
    b = ajouter_bruit_poivre_sel(image, 20, 30, seed=7)
    assert np.array_equal(a, b)
    assert np.count_nonzero(a == 255) == 20
    assert np.count_nonzero(a == 0) == 30


def test_filtres_et_psnr() -> None:
    image = np.arange(81, dtype=np.uint8).reshape(9, 9)
    for resultat in (
        filtre_moyenneur_blur(image, 3),
        filtre_gaussien(image, taille=3),
        filtre_median(image, 3),
    ):
        assert resultat.shape == image.shape
        assert resultat.dtype == np.uint8
    assert psnr(image, image) == float("inf")
