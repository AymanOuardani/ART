"""
Trace l'évolution de l'erreur pendant l'entraînement LOSO d'un sujet : la courbe
d'entraînement face à celle de validation, avec le minimum repéré en rouge. Un écart qui se
creuse entre les deux signale du surapprentissage ; une courbe plate, un plafond atteint.

  python mse_graph.py 4                      sujet 4
  python mse_graph.py 4 --modele ART_v2      autre dossier d'entraînement dans Model/

Les données viennent de Model/<modèle>/resultats_LOSO.xlsx, écrit par Train_Model.py.
"""

import argparse as ap
import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Chemins des fichiers
Model_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Model")
Modele_Defaut = "ART_ICLABEL"                      # dossier des checkpoints LOSO dans Model/

#Ligne de commande pour récupérer le numéro du sujet
parser = ap.ArgumentParser(description="Évolution de l'erreur (train/validation) par epoch, sujet exclu (LOSO)")
parser.add_argument("Sujet", type=int, help="Numéro du sujet exclu (1-109)")
parser.add_argument("--modele", default=Modele_Defaut,
                    help=f"dossier de l'entraînement dans Model/ (défaut : {Modele_Defaut})")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)
Fichier_Excel = Model_Dir / args.modele / "resultats_LOSO.xlsx"

if not Fichier_Excel.exists():
    raise SystemExit(f"ERREUR : fichier introuvable : {Fichier_Excel}\n"
                     f"  (lance d'abord Train_Model.py)")

feuilles = pd.read_excel(Fichier_Excel, sheet_name=None, engine="openpyxl")
if sujet_id not in feuilles:
    raise SystemExit(f"ERREUR : aucune donnée pour {sujet_id} dans {Fichier_Excel.name}\n"
                     f"  Sujets disponibles : {', '.join(feuilles)}")
df = feuilles[sujet_id]

#Courbes globales : train (moyenne/epoch) vs validation, par epoch
#Feuilles récentes : RMSE en µV (Train_Model.py actuel). Anciennes feuilles : MSE sur signal
#normalisé, sans unité (ancien protocole) -> on garde le repli pour pouvoir les relire.
if "epoch_train_rmse" in df.columns:
    courbe_train = df["epoch_train_rmse"].dropna().values
    courbe_val = df["epoch_val_rmse"].dropna().values
    unite, nom_courbe, decimales = "RMSE (µV)", "RMSE de validation", 2
else:
    courbe_train = df["epoch_train_mse"].dropna().values
    courbe_val = df["epoch_eval_mse"].dropna().values
    unite, nom_courbe, decimales = "MSE", "MSE d'évaluation", 3

epochs = np.arange(1, len(courbe_train) + 1)

#Point minimal de la courbe de validation (meilleur epoch)
best_idx = int(np.argmin(courbe_val))
best_epoch = epochs[best_idx]
best_val = courbe_val[best_idx]

#Graphique : évolution par epoch (train vs validation)
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

ax.set_title(f"{sujet_id} — {nom_courbe} minimal "
             f"{best_val:.{decimales}f}{' µV' if decimales == 2 else ''} à l'epoch {best_epoch}")
fig.tight_layout()

print(f"{sujet_id} : {nom_courbe} minimal "
      f"{best_val:.{decimales}f}{' µV' if decimales == 2 else ''} à l'epoch {best_epoch} "
      f"(sur {len(courbe_val)} epochs)")

plt.show()
