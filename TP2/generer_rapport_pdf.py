"""Génère le compte rendu PDF professionnel du TP2."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


RACINE = Path(__file__).resolve().parent
SORTIE = RACINE / "Compte_rendu_TP2_Anwar_Ezzirani.pdf"


def polices() -> tuple[str, str, str]:
    arial = Path("C:/Windows/Fonts/arial.ttf")
    arial_bold = Path("C:/Windows/Fonts/arialbd.ttf")
    consolas = Path("C:/Windows/Fonts/consola.ttf")
    if arial.exists() and arial_bold.exists():
        pdfmetrics.registerFont(TTFont("Rapport", arial))
        pdfmetrics.registerFont(TTFont("Rapport-Bold", arial_bold))
        if consolas.exists():
            pdfmetrics.registerFont(TTFont("Rapport-Code", consolas))
            return "Rapport", "Rapport-Bold", "Rapport-Code"
        return "Rapport", "Rapport-Bold", "Courier"
    return "Helvetica", "Helvetica-Bold", "Courier"


NORMAL, GRAS, CODE = polices()


def pied_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(1.8 * cm, 1.35 * cm, A4[0] - 1.8 * cm, 1.35 * cm)
    canvas.setFont(NORMAL, 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(1.8 * cm, 0.98 * cm, "Vision Artificielle — TP n°2")
    canvas.drawRightString(A4[0] - 1.8 * cm, 0.98 * cm, f"Page {doc.page}")
    canvas.restoreState()


def styles_rapport():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitreTP", fontName=GRAS, fontSize=24, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#17365D"), spaceAfter=15))
    styles.add(ParagraphStyle(name="SousTitreTP", fontName=NORMAL, fontSize=13, leading=19, alignment=TA_CENTER, textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle(name="SectionTP", fontName=GRAS, fontSize=15, leading=19, textColor=colors.HexColor("#17365D"), spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="SousSectionTP", fontName=GRAS, fontSize=11.5, leading=15, textColor=colors.HexColor("#1D4ED8"), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="CorpsTP", fontName=NORMAL, fontSize=9.7, leading=13.5, alignment=TA_JUSTIFY, textColor=colors.HexColor("#1F2937"), spaceAfter=6))
    styles.add(ParagraphStyle(name="LegendeTP", fontName=NORMAL, fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceBefore=3, spaceAfter=7))
    styles.add(ParagraphStyle(name="CodeTP", fontName=CODE, fontSize=7, leading=9, leftIndent=6, rightIndent=6, borderColor=colors.HexColor("#CBD5E1"), borderWidth=0.5, borderPadding=6, backColor=colors.HexColor("#F8FAFC"), spaceBefore=3, spaceAfter=7))
    return styles


def bloc_code(texte: str, styles) -> Preformatted:
    return Preformatted(texte.strip(), styles["CodeTP"])


def figure(nom: str, largeur: float, hauteur: float, legende: str, styles):
    chemin = RACINE / "resultats" / nom
    if not chemin.exists():
        return []
    return [Image(str(chemin), width=largeur, height=hauteur), Paragraph(legende, styles["LegendeTP"])]


def generer() -> None:
    styles = styles_rapport()
    corps = styles["CorpsTP"]
    doc = SimpleDocTemplate(
        str(SORTIE), pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.5 * cm, bottomMargin=1.8 * cm,
        title="Compte rendu TP2 — Filtrage d'images",
        author="Anwar Ezzirani — G4",
    )
    story = []

    story.extend([
        Spacer(1, 2.3 * cm),
        Paragraph("VISION ARTIFICIELLE", styles["SousTitreTP"]),
        Spacer(1, 0.9 * cm),
        Paragraph("TP n°2", styles["TitreTP"]),
        Paragraph("Filtrage d’images", styles["TitreTP"]),
        Spacer(1, 1.3 * cm),
    ])
    infos = Table([
        ["Établissement", "EMSI Casa"],
        ["Étudiant", "Anwar Ezzirani"],
        ["Groupe", "G4"],
        ["Année universitaire", "2026–2027"],
        ["Technologies", "Python, NumPy, Matplotlib, SciPy / OpenCV"],
    ], colWidths=[5.0 * cm, 10.0 * cm], rowHeights=0.82 * cm)
    infos.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), NORMAL),
        ("FONTNAME", (0, 0), (0, -1), GRAS),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E2E8F0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([infos, Spacer(1, 1.7 * cm), Paragraph("Compte rendu et résultats expérimentaux", styles["SousTitreTP"]), PageBreak()])

    story.append(Paragraph("1. Objectifs", styles["SectionTP"]))
    story.append(Paragraph(
        "Ce TP étudie le filtrage spatial des images. Il consiste à programmer une convolution avec padding, "
        "appliquer différents noyaux, générer un bruit poivre et sel, puis comparer les filtres moyenneur, "
        "gaussien et médian. Le PSNR permet enfin d’évaluer quantitativement la qualité des restaurations.", corps))
    story.append(Paragraph(
        "Le fichier aero1.jpg n’étant pas joint au sujet, l’expérience utilise une image naturelle fournie avec "
        "Matplotlib. Le programme utilisera automatiquement aero1.jpg dès qu’il sera placé dans le dossier.", corps))

    story.append(Paragraph("2. Convolution", styles["SectionTP"]))
    story.append(Paragraph(
        "Le noyau est déplacé sur chaque pixel. Les produits entre le voisinage et le noyau sont additionnés "
        "séparément pour chaque canal RGB. Un zero-padding conserve les dimensions et les résultats sont "
        "arrondis puis saturés dans l’intervalle [0, 255].", corps))
    story.append(bloc_code("""
for i in range(hauteur):
    for j in range(largeur):
        voisinage = image_pad[i:i+h, j:j+w]
        somme = np.sum(voisinage * noyau[..., None], axis=(0, 1))
        resultat[i, j] = np.rint(somme).clip(0, 255)
""", styles))

    story.append(Paragraph("3. Noyaux demandés", styles["SectionTP"]))
    noyaux = Table([
        ["Effet", "Noyau / principe"],
        ["Lissage", "Matrice 3×3 remplie de 1/9 : moyenne du voisinage"],
        ["Contraste", "Centre = 5 et quatre voisins directs = −1"],
        ["Repoussage", "Coefficients orientés de −2 à 2 : effet de relief"],
    ], colWidths=[3.4 * cm, 12.0 * cm])
    noyaux.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), NORMAL),
        ("FONTNAME", (0, 0), (-1, 0), GRAS),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(noyaux)
    story.extend(figure("01_noyaux_convolution.png", 15.4 * cm, 13.0 * cm, "Figure 1 — Image originale et résultats des trois noyaux.", styles))

    story.append(PageBreak())
    story.append(Paragraph("4. Ajout du bruit poivre et sel", styles["SectionTP"]))
    story.append(Paragraph(
        "Le programme choisit aléatoirement entre 300 et 10 000 pixels blancs et noirs. Sans graine fixe, "
        "les positions changent à chaque exécution. Pour rendre ce rapport reproductible, la graine 42 a été "
        "utilisée : 1 165 pixels sel et 7 808 pixels poivre ont été ajoutés. L’histogramme bruité présente "
        "les pics caractéristiques aux intensités 0 et 255.", corps))
    story.extend(figure("02_bruit_histogramme.png", 16.5 * cm, 12.0 * cm, "Figure 2 — Image originale, bruit impulsionnel et histogrammes.", styles))

    story.append(Paragraph("5. Filtre moyenneur", styles["SectionTP"]))
    story.append(Paragraph(
        "Les fenêtres 3×3, 5×5 et 7×7 correspondent respectivement à des noyaux constants divisés par 9, "
        "25 et 49. Les méthodes filter2D et blur donnent le même résultat. Une fenêtre plus grande retire "
        "davantage de bruit mais augmente le flou et réduit le PSNR.", corps))
    story.append(bloc_code("""
kernel_5x5 = np.ones((5, 5), np.float32) / 25
filtered_5x5 = cv2.filter2D(img, -1, kernel_5x5)
# Méthode équivalente :
filtered_5x5 = cv2.blur(img, (5, 5))
""", styles))
    story.extend(figure("03_filtres_moyenneurs.png", 16.5 * cm, 4.7 * cm, "Figure 3 — Effet de la taille du filtre moyenneur.", styles))

    story.append(PageBreak())
    story.append(Paragraph("6. Filtre gaussien", styles["SectionTP"]))
    story.append(Paragraph(
        "Le filtre gaussien accorde davantage de poids aux pixels proches du centre. Il préserve donc mieux "
        "les structures qu’une moyenne uniforme. Les tailles 3×3, 5×5 et 7×7 ont été comparées, puis les "
        "valeurs de sigma 1, 5 et 10 ont été testées. Les grands sigma produisent ici un flou excessif.", corps))
    story.extend(figure("04_moyenneur_vs_gaussien.png", 15.6 * cm, 9.6 * cm, "Figure 4 — Comparaison des filtres moyenneurs et gaussiens.", styles))
    story.extend(figure("05_gaussien_sigmas.png", 16.5 * cm, 4.8 * cm, "Figure 5 — Influence de sigma : 1, 5 et 10.", styles))

    story.append(PageBreak())
    story.append(Paragraph("7. Filtre médian", styles["SectionTP"]))
    story.append(Paragraph(
        "Le filtre médian est particulièrement efficace contre le bruit poivre et sel : les valeurs extrêmes "
        "0 et 255 ont peu d’influence sur la médiane du voisinage. Le médian 3×3 atteint 30,01 dB, soit le "
        "meilleur PSNR de cette expérience.", corps))
    story.extend(figure("06_filtre_median_poivre_sel.png", 15.3 * cm, 11.8 * cm, "Figure 6 — Comparaison sur le bruit poivre et sel.", styles))

    story.append(Paragraph("8. Test sur un bruit gaussien", styles["SectionTP"]))
    story.append(Paragraph(
        "Sur un bruit gaussien d’écart-type 25, le filtre médian améliore l’image mais le filtre gaussien "
        "obtient un meilleur PSNR : 26,05 dB contre 25,20 dB. Il est donc préférable d’adapter le filtre "
        "à la nature du bruit.", corps))
    story.extend(figure("07_filtre_median_bruit_gaussien.png", 16.5 * cm, 4.9 * cm, "Figure 7 — Résultats sur un bruit gaussien.", styles))

    story.append(PageBreak())
    story.append(Paragraph("9. Résultats quantitatifs", styles["SectionTP"]))
    donnees = [
        ["Traitement", "PSNR (dB)"],
        ["Image avec bruit poivre et sel", "20,71"],
        ["Moyenneur 3×3", "25,88"], ["Moyenneur 5×5", "23,89"], ["Moyenneur 7×7", "22,39"],
        ["Gaussien 3×3", "26,39"], ["Gaussien 5×5", "25,86"], ["Gaussien 7×7", "24,88"],
        ["Médian 3×3", "30,01"], ["Médian 5×5", "26,46"], ["Médian 7×7", "24,58"],
    ]
    tableau = Table(donnees, colWidths=[11.8 * cm, 3.4 * cm])
    tableau.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), NORMAL),
        ("FONTNAME", (0, 0), (-1, 0), GRAS),
        ("FONTNAME", (0, 8), (-1, 8), GRAS),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 8), (-1, 8), colors.HexColor("#DCFCE7")),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94A3B8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, 7), [colors.white, colors.HexColor("#F8FAFC")]),
        ("ROWBACKGROUNDS", (0, 9), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tableau)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Le PSNR vaut 10 log10(255² / MSE). Une valeur élevée signifie que l’image filtrée est plus proche "
        "de l’image originale. Le meilleur résultat contre le bruit poivre et sel est obtenu par le filtre "
        "médian 3×3.", corps))

    story.append(Paragraph("10. Conclusion", styles["SectionTP"]))
    story.append(Paragraph(
        "La convolution permet de construire de nombreux traitements à partir d’une petite matrice. Le "
        "filtre moyenneur est simple mais floute rapidement les contours. Le filtre gaussien préserve mieux "
        "les structures et convient au bruit gaussien. Le filtre médian est le plus efficace contre le bruit "
        "impulsionnel. Le choix du filtre et de sa taille constitue donc un compromis entre suppression du "
        "bruit et conservation des détails.", corps))
    story.append(Paragraph(
        "Les cinq tests fonctionnels du programme ont réussi et les deux méthodes du filtre moyenneur ont "
        "présenté un écart maximal nul pour les tailles 3×3, 5×5 et 7×7.", corps))

    doc.build(story, onFirstPage=pied_page, onLaterPages=pied_page)
    print(f"Rapport créé : {SORTIE}")


if __name__ == "__main__":
    generer()
