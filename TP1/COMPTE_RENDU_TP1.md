# TP n°1 — Introduction au traitement d’images

**Module :** Vision Artificielle — EMSI Casa  
**Étudiant :** Anwar Ezzirani  
**Groupe :** 4  
**Image de test :** `Lenna_(test_image).png`

## Objectif

Ce TP présente la représentation matricielle d’une image numérique et les premières opérations de traitement d’images avec Python, NumPy et Matplotlib : transformation géométrique, négatif, luminance, séparation des canaux RGB et égalisation d’histogramme.

Le code complet se trouve dans `TP1_solution.py`. Il conserve volontairement les boucles demandées dans l’énoncé afin de montrer le traitement pixel par pixel.

## Réponses

### Question 1 — Symétrie verticale

Pour une image de largeur `p`, le pixel de coordonnées `(i, j)` est remplacé par le pixel `(i, p - 1 - j)`. La ligne ne change donc pas, tandis que l’ordre des colonnes est inversé.

```python
def symetrie(im):
    n, p, c = im.shape
    im_sym = np.zeros_like(im)
    for i in range(n):
        for j in range(p):
            im_sym[i, j] = im[i, p - 1 - j]
    return im_sym
```

### Question 2 — Négatif

Chaque composante `v` dans `[0, 255]` est remplacée par son complément `255 - v`.

```python
def negatif(im):
    n, p, c = im.shape
    im_neg = np.zeros_like(im)
    for i in range(n):
        for j in range(p):
            im_neg[i, j] = 255 - im[i, j]
    return im_neg
```

### Question 4 — Conversion en niveaux de gris

La luminance respecte la formule du sujet :

`y = 0,2126 r + 0,7152 g + 0,0722 b`

```python
def niveaudegris(im):
    n, p, c = im.shape
    im_lum = np.zeros((n, p), dtype=np.uint8)
    for i in range(n):
        for j in range(p):
            r, g, b = im[i, j].astype(float)
            im_lum[i, j] = np.uint8(np.rint(
                0.2126 * r + 0.7152 * g + 0.0722 * b
            ))
    return im_lum
```

Le coefficient du vert est le plus élevé, car l’œil humain y est plus sensible.

### Questions 5, 6 et 7 — Canaux RGB

Une nouvelle image de même dimension est créée avec des zéros, puis seul le canal souhaité est recopié : indice `0` pour le rouge, `1` pour le vert et `2` pour le bleu.

```python
def image_rouge(im):
    resultat = np.zeros_like(im)
    resultat[:, :, 0] = im[:, :, 0]
    return resultat

def image_verte(im):
    resultat = np.zeros_like(im)
    resultat[:, :, 1] = im[:, :, 1]
    return resultat

def image_bleue(im):
    resultat = np.zeros_like(im)
    resultat[:, :, 2] = im[:, :, 2]
    return resultat
```

### Question 8 — Histogramme

L’histogramme contient 256 cases. Pour chaque pixel, la case correspondant à son niveau de gris est incrémentée. La somme de toutes les cases doit être égale au nombre total de pixels.

```python
def histo(im):
    H = np.zeros(256, dtype=np.int64)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            H[int(im[i, j])] += 1
    return H
```

### Question 9 — Table d’égalisation

On calcule d’abord l’histogramme cumulé `hc`, puis on le normalise dans l’intervalle `[0, 255]`. Le tableau `he` obtenu sert de table de conversion.

```python
def egaliser_histogramme(H, gray_image):
    hc = np.zeros(len(H))
    hc[0] = H[0]
    for i in range(1, len(H)):
        hc[i] = hc[i - 1] + H[i]
    he = (hc / gray_image.size) * 255
    return he
```

### Question 10 — Augmentation du contraste

Chaque niveau de gris original devient le niveau indiqué par la table d’égalisation. L’histogramme résultant est davantage réparti sur la plage disponible, ce qui améliore généralement le contraste global.

```python
def augmenter_contraste(gray_image):
    H = histo(gray_image)
    he = egaliser_histogramme(H, gray_image)
    table = np.rint(he).clip(0, 255).astype(np.uint8)
    new_image = gray_image.copy()
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            new_image[i, j] = table[gray_image[i, j]]
    return new_image
```

## Exécution

Depuis le dossier du TP :

```powershell
python TP1_solution.py
```

Le programme crée automatiquement le dossier `resultats`, les sept images transformées, une planche comparative et la comparaison des histogrammes avant/après égalisation.

## Conclusion

Ce TP montre qu’une image couleur est un tableau tridimensionnel et qu’une image en niveaux de gris est une matrice 2D. Les transformations simples s’expriment comme des opérations sur les coordonnées ou les composantes des pixels. L’égalisation exploite quant à elle la distribution globale des intensités pour utiliser plus efficacement la dynamique `[0, 255]`.
