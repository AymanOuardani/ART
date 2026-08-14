"""
Trace l'erreur d'entraînement et de validation epoch par epoch, avec le minimum de
validation repéré en rouge. Un écart qui se creuse signale du surapprentissage.

  python MSE_Plot.py ART_Local 4    sujet 4
  python MSE_Plot.py DuoCL
  python MSE_Plot.py GCTNet

Les données viennent des classeurs écrits par Train_Model.py : Ressources/Train_ART_Local.xlsx
pour ART_Local, une feuille par sujet ; Ressources/Train_DuoGCT.xlsx pour les deux autres, une
feuille par modèle et par bruit.
"""

"""
English summary: plots the per-epoch training and validation error curves for a given
trained model, marking the validation minimum in red (a widening train/val gap points
to overfitting). Reads results previously written by Train_Model.py.

Usage:
  python MSE_Plot.py <model> [subject]
  <model> is one of: ART_Local, DuoCL, GCTNet. <subject> (excluded subject number) is
  required only for ART_Local.
"""

import argparse as ap
import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Chemins des fichiers
Racine = pl.Path(__file__).resolve().parent
Output = Racine / "Output"
Ressources = Racine / "Ressources"
Excel_ART = Ressources / "Train_ART_Local.xlsx"
Excel_DuoGCT = Ressources / "Train_DuoGCT.xlsx"

#Ligne de commande
parser = ap.ArgumentParser(description="Erreur par epoch d'un entraînement")
parser.add_argument("Modele", choices=["ART_Local", "DuoCL", "GCTNet"], help="modèle entraîné")
parser.add_argument("Sujet", type=int, nargs="?", default=None,
                    help="numéro du sujet exclu (requis pour ART_Local, ex. 4)")
args = parser.parse_args()

if args.Modele == "ART_Local" and args.Sujet is None:
    raise SystemExit("ERREUR : ART_Local nécessite un numéro de sujet, ex. "
                     "python MSE_Plot.py ART_Local 4")

fichier = Excel_ART if args.Modele == "ART_Local" else Excel_DuoGCT
if not fichier.exists():
    raise SystemExit(f"ERREUR : fichier introuvable : {fichier}\n"
                     f"  (lance d'abord Train_Model.py {args.Modele})")
feuilles = pd.read_excel(fichier, sheet_name=None, engine="openpyxl")

#ART_Local a une feuille par sujet et des courbes en µV, les deux autres une feuille par
#bruit et un MSE sans unité
if args.Modele == "ART_Local":
    feuille = "S" + str(args.Sujet).zfill(3)
    if feuille not in feuilles:
        raise SystemExit(f"ERREUR : aucune donnée pour {feuille} dans {fichier.name}\n"
                         f"  Sujets disponibles : {', '.join(feuilles)}")
    df = feuilles[feuille]
    courbe_train = df["epoch_train_rmse"].dropna().values
    courbe_val = df["epoch_val_rmse"].dropna().values
    unite, nom_courbe, decimales, suffixe = "RMSE (µV)", "RMSE de validation", 2, " µV"
else:
    noms = [n for n in feuilles if n.startswith(args.Modele)]
    if not noms:
        raise SystemExit(f"ERREUR : aucune donnée pour {args.Modele} dans {fichier.name}\n"
                         f"  Feuilles disponibles : {', '.join(feuilles)}")
    feuille = noms[0]
    df = feuilles[feuille]
    courbe_train = df["train_mse"].dropna().values
    courbe_val = df["val_mse"].dropna().values
    unite, nom_courbe, decimales, suffixe = "MSE", "MSE de validation", 3, ""

epochs = np.arange(1, len(courbe_train) + 1)

#Minimum de la validation
best_idx = int(np.argmin(courbe_val))
best_epoch = epochs[best_idx]
best_val = courbe_val[best_idx]

#Graphique
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.plot(epochs, courbe_train, label="Train")
ax.plot(epochs, courbe_val, label="Validation")
ymin, ymax = ax.get_ylim()
ax.scatter([best_epoch], [best_val], color="red", zorder=5)
ax.vlines(best_epoch, ymin, best_val, color="red", linestyle="--", linewidth=1)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Epoch")
ax.set_ylabel(unite)
ax.legend()
ax.grid(alpha=0.3)

ax.set_title(f"{feuille} — {nom_courbe} minimal "
             f"{best_val:.{decimales}f}{suffixe} à l'epoch {best_epoch}")
fig.tight_layout()

print(f"{feuille} : {nom_courbe} minimal {best_val:.{decimales}f}{suffixe} "
      f"à l'epoch {best_epoch} (sur {len(courbe_val)} epochs)")

plt.show()
