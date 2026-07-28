import argparse as ap
import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utils

#Chemins des fichiers
Fichier_Excel = pl.Path(r"C:\Users\aymen\Desktop\ART\Ressources\resultats_LOSO.xlsx")
Rapport_Tex = pl.Path(r"C:\Users\aymen\Desktop\ART\Résultats\Rapport_ART.tex")
Images_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Résultats\Images")

#Ligne de commande pour récupérer le numéro du sujet
parser = ap.ArgumentParser(description="Évolution du MSE (train/eval) par epoch, sujet exclu (LOSO)")
parser.add_argument("Sujet", type=int, help="Numéro du sujet exclu (1-109)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

if not Fichier_Excel.exists():
    raise SystemExit(f"ERREUR : fichier introuvable : {Fichier_Excel}\n"
                     f"  (lance d'abord Train_ART.py, puis copie resultats_LOSO.xlsx dans Ressources/)")

feuilles = pd.read_excel(Fichier_Excel, sheet_name=None, engine="openpyxl")
if sujet_id not in feuilles:
    raise SystemExit(f"ERREUR : aucune donnée pour {sujet_id} dans {Fichier_Excel.name}\n"
                     f"  Sujets disponibles : {', '.join(feuilles)}")
df = feuilles[sujet_id]

#Courbes globales : MSE train (moyenne/epoch) vs MSE eval (sujet exclu), par epoch
mse_train = df["epoch_train_mse"].dropna().values
mse_eval = df["epoch_eval_mse"].dropna().values
epochs = np.arange(1, len(mse_train) + 1)

#Détail local : colonnes batch_epoch_N mises bout à bout, avec repères de fin d'epoch
colonnes_batch = sorted([c for c in df.columns if c.startswith("batch_epoch_")],
                        key=lambda c: int(c.split("_")[-1]))
pertes_batch, limites_epoch = [], [0]
for c in colonnes_batch:
    vals = df[c].dropna().values
    pertes_batch.extend(vals)
    limites_epoch.append(limites_epoch[-1] + len(vals))

#Graphique : évolution globale (haut) + détail batch par batch (bas)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

ax1.plot(epochs, mse_train, marker="o", label="MSE train (moyenne/epoch)")
ax1.plot(epochs, mse_eval, marker="o", label="MSE eval (sujet exclu)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.set_title(f"{sujet_id} — évolution du MSE par epoch (LOSO)")
ax1.legend()
ax1.grid(alpha=0.3)

#Lissage (moyenne glissante) : avec ~1000+ batchs/epoch, la courbe brute est illisible
fenetre = 30
pertes_lissees = pd.Series(pertes_batch).rolling(fenetre, min_periods=1, center=True).mean()

ax2.plot(range(1, len(pertes_batch) + 1), pertes_batch, linewidth=0.4, color="tab:gray",
         alpha=0.3, label="perte train (par batch, brut)")
ax2.plot(range(1, len(pertes_batch) + 1), pertes_lissees, linewidth=1.2, color="tab:blue",
         label=f"perte train (moyenne glissante, {fenetre} batchs)")
for lim in limites_epoch[1:-1]:
    ax2.axvline(lim, color="black", linewidth=0.4, alpha=0.15)
ax2.set_xlabel("Batch (toutes epochs mises bout à bout)")
ax2.set_ylabel("Perte (batch)")
ax2.set_title(f"{sujet_id} — détail batch par batch")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()

#Sauvegarde dans Résultats/Images/ + mise à jour du rapport (si le dépôt Résultats existe)
if Rapport_Tex.exists():
    Images_Dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(Images_Dir / f"{sujet_id}_mse.png", dpi=150)
    print("Figure sauvegardée :", Images_Dir / f"{sujet_id}_mse.png")
    Utils.recompile_latex(Rapport_Tex)

plt.show()
