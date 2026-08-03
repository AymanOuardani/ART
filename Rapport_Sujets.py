"""
Enchaîne toute la chaîne du rapport pour une plage de sujets, avec une seule compilation
LaTeX à la fin (au lieu d'une par script, soit une centaine d'appels inutiles à pdflatex).

  python Rapport_Sujets.py 1-16
  python Rapport_Sujets.py 1-16 --modele ART_ICLABEL_v2   # autre dossier d'entraînement
  python Rapport_Sujets.py 1-16 --sans-art                # sans les 60 epochs (rapide)

Pour chaque sujet : les 60 epochs d'ART (tableau + figures), l'accuracy d'ART_Orig et
d'ICLabel, le RMSE des six méthodes contre ICLabel, et la courbe train/validation.
Puis le tableau récapitulatif de la meilleure epoch et la compilation du rapport.
"""

import argparse as ap
import os
import pathlib as pl
import subprocess
import sys
import time

RACINE = pl.Path(__file__).resolve().parent

#Méthodes comparées à ICLabel dans le tableau RMSE de chaque sujet (cf. mse_eval.py)
METHODES_RMSE = ["ART", "ICUNet", "ICUNet++", "ICUNet_attn", "DuoCL", "GCTNet"]

parser = ap.ArgumentParser(description="Régénère les pages du rapport pour une plage de sujets")
parser.add_argument("Sujets", nargs="?", default="1-109", help="ex. 1-16 ou 1,2,5 (défaut : 1-109)")
parser.add_argument("--modele", default="ART_ICLABEL",
                    help="dossier de l'entraînement dans Model/ (défaut : ART_ICLABEL)")
parser.add_argument("--sans-art", action="store_true",
                    help="ne recalcule pas les 60 epochs d'ART (seulement accuracy, RMSE et courbes)")
parser.add_argument("--force", action="store_true",
                    help="recalcule les epochs d'ART déjà enregistrées")
args = parser.parse_args()

if "-" in args.Sujets:
    a, b = args.Sujets.split("-")
    sujets = list(range(int(a), int(b) + 1))
else:
    sujets = [int(s) for s in args.Sujets.split(",")]

#Aucune fenêtre matplotlib : le script peut tourner sans surveillance
env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")


def etape(*commande):
    # Lance un script du dépôt ; un sujet sans fichier ne doit pas interrompre toute la série
    r = subprocess.run([sys.executable, *commande], cwd=RACINE, env=env)
    if r.returncode != 0:
        print(f"  (échec : {' '.join(str(c) for c in commande)})", flush=True)
    return r.returncode == 0


debut = time.time()
for s in sujets:
    print(f"\n########## S{s:03d} ##########", flush=True)

    if not args.sans_art:
        cmd = ["ART_Epochs.py", str(s), "--sans-latex", "--modele", args.modele]
        etape(*(cmd + ["--force"] if args.force else cmd))

    #Accuracy : ART d'origine (colonne ART_Orig) puis ICLabel, la référence
    etape("Evaluation.py", "ART", str(s), "--sans-latex")
    etape("Evaluation.py", "ICLABEL", str(s), "--sans-latex")

    #RMSE de chaque méthode contre ICLabel
    for m in METHODES_RMSE:
        etape("mse_eval.py", m, str(s), "--sans-latex")

    #Courbe train/validation de l'entraînement LOSO du sujet
    etape("mse_graph.py", str(s), "--sans-latex", "--modele", args.modele)

#Tableau récapitulatif de la meilleure epoch, qui recompile le rapport une seule fois
print("\n########## récapitulatif ##########", flush=True)
etape("Recap_ART.py", args.Sujets)

print(f"\nTerminé : {len(sujets)} sujets en {(time.time() - debut) / 60:.0f} min", flush=True)
