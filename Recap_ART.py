import argparse as ap
import json
import pathlib as pl

import numpy as np

import Utils

#Chemins des fichiers
Rapport_Tex = pl.Path(r"C:\Users\aymen\Desktop\ART\Résultats\Rapport_ART.tex")
Data_Dir = Rapport_Tex.parent / "data"

#Ligne de commande : quels sujets figurent dans le tableau
parser = ap.ArgumentParser(description="Tableau récapitulatif d'ART à la meilleure epoch de validation")
parser.add_argument("Sujets", nargs="?", default="1-109", help="ex. 1-16 ou 1,2,5 (défaut : 1-109)")
parser.add_argument("--sans-latex", action="store_true", help="écrit le tableau sans lancer pdflatex")
args = parser.parse_args()

if "-" in args.Sujets:
    a, b = args.Sujets.split("-")
    sujets = list(range(int(a), int(b) + 1))
else:
    sujets = [int(s) for s in args.Sujets.split(",")]


def cellule(fichier, cle):
    # Récupère la valeur d'une ligne déjà écrite par Evaluation.py : "ICLabel & 0.64 $\pm$ 0.23 \\"
    # -> "0.64 $\pm$ 0.23" (même découpage que Utils.genere_recap)
    if not fichier.exists():
        return None
    data = json.loads(fichier.read_text(encoding="utf-8"))
    if cle not in data:
        return None
    return data[cle].split("&", 1)[1].rsplit("\\\\", 1)[0].strip()


def moyenne_accuracy(valeurs):
    # "0.68 $\pm$ 0.18" x N -> moyenne des moyennes et moyenne des écarts-types
    moys, ecarts = [], []
    for v in valeurs:
        if v is None:
            continue
        try:
            m, e = v.split("$\\pm$")
            moys.append(float(m))
            ecarts.append(float(e))
        except ValueError:
            continue
    if not moys:
        return None
    return f"{np.mean(moys):.2f} $\\pm$ {np.mean(ecarts):.2f}"


lignes = []
colonnes_num = {"rmse": [], "snr": []}
colonnes_acc = {"art": [], "orig": [], "iclabel": []}

for s in sujets:
    sujet_id = "S" + str(s).zfill(3)
    f_epochs = Data_Dir / f"{sujet_id}_epochs.json"
    if not f_epochs.exists():
        continue
    etat = json.loads(f_epochs.read_text(encoding="utf-8"))
    if "meilleure" not in etat:       # sujet en cours de traitement
        continue
    best = etat["meilleure"]

    f_acc = Data_Dir / f"{sujet_id}_accuracy.json"
    acc_art = f"{best['acc']:.2f} $\\pm$ {best['std']:.2f}"
    acc_orig = cellule(f_acc, "ART")           # modèle ART d'origine, non réentraîné
    acc_iclabel = cellule(f_acc, "ICLABEL")

    lignes.append(f"{sujet_id} & {best['epoch']} & {best['rmse']:.2f} & {best['snr']:.2f} & "
                  f"{acc_art} & {acc_orig or '-'} & {acc_iclabel or '-'} \\\\")

    for cle in colonnes_num:
        colonnes_num[cle].append(best[cle])
    colonnes_acc["art"].append(acc_art)
    colonnes_acc["orig"].append(acc_orig)
    colonnes_acc["iclabel"].append(acc_iclabel)

if not lignes:
    raise SystemExit("ERREUR : aucun sujet exploitable (lance d'abord ART_Epochs.py)")

#Dernière ligne : moyenne sur les sujets présents, en gras comme les autres récapitulatifs
moyennes = [f"\\textbf{{{np.mean(colonnes_num[c]):.2f}}}" for c in ("rmse", "snr")]
moyennes_acc = [moyenne_accuracy(colonnes_acc[c]) for c in ("art", "orig", "iclabel")]
lignes.append(r"\midrule \textbf{Moyenne} & --- & " + " & ".join(moyennes) + " & "
              + " & ".join(f"\\textbf{{{m}}}" if m else "-" for m in moyennes_acc) + r" \\")

sortie = Data_Dir / "recap_art_meilleure_epoch.tex"
sortie.write_text("\\def\\recapartmeilleureepoch{\n" + "\n".join(lignes) + "}\n",
                  encoding="utf-8", newline="\n")
print(f"Tableau écrit ({len(lignes) - 1} sujets) : {sortie}")

if Rapport_Tex.exists() and not args.sans_latex:
    Utils.recompile_latex(Rapport_Tex)
