import argparse as ap
import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utils

#Chemins des fichiers
Model_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Model")
Modele_Defaut = "ART_ICLABEL"                      # dossier des checkpoints LOSO dans Model/
Rapport_Tex = pl.Path(r"C:\Users\aymen\Desktop\ART\Résultats\Rapport_ART.tex")
Images_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Résultats\Images")

#Ligne de commande pour récupérer le numéro du sujet
parser = ap.ArgumentParser(description="Évolution de l'erreur (train/validation) par epoch, sujet exclu (LOSO)")
parser.add_argument("Sujet", type=int, help="Numéro du sujet exclu (1-109)")
parser.add_argument("--modele", default=Modele_Defaut,
                    help=f"dossier de l'entraînement dans Model/ (défaut : {Modele_Defaut})")
parser.add_argument("--sans-latex", action="store_true",
                    help="écrit la figure sans ouvrir de fenêtre ni lancer pdflatex (traitement en série)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)
Fichier_Excel = Model_Dir / args.modele / "resultats_LOSO.xlsx"

if not Fichier_Excel.exists():
    raise SystemExit(f"ERREUR : fichier introuvable : {Fichier_Excel}\n"
                     f"  (lance d'abord Train_ART.py)")

feuilles = pd.read_excel(Fichier_Excel, sheet_name=None, engine="openpyxl")
if sujet_id not in feuilles:
    raise SystemExit(f"ERREUR : aucune donnée pour {sujet_id} dans {Fichier_Excel.name}\n"
                     f"  Sujets disponibles : {', '.join(feuilles)}")
df = feuilles[sujet_id]

#Courbes globales : train (moyenne/epoch) vs validation, par epoch
#Feuilles récentes : RMSE en µV (Train_ART.py actuel). Anciennes feuilles : MSE sur signal
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

fig.tight_layout()

#Sauvegarde dans Résultats/Images/ + mise à jour du rapport (si le dépôt Résultats existe)
if Rapport_Tex.exists():
    Images_Dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(Images_Dir / f"{sujet_id}_mse.png", dpi=150)
    print("Figure sauvegardée :", Images_Dir / f"{sujet_id}_mse.png")

    data_dir = Rapport_Tex.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{sujet_id}_mse_sentence.tex").write_text(
        "\\def\\msesentence{Le " + nom_courbe + " minimal "
        f"({best_val:.{decimales}f}{' µV' if decimales == 2 else ''}) "
        f"se présente à l'epoch {best_epoch}.}}\n", encoding="utf-8")

    if not args.sans_latex:
        Utils.recompile_latex(Rapport_Tex)

#En série (--sans-latex), la figure est déjà écrite : ouvrir une fenêtre bloquerait le script
if not args.sans_latex:
    plt.show()
