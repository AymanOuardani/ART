"""
Décodage main gauche vs main droite par CSP + LDA, en validation croisée (10 tirages 80/20).

  python Evaluation.py brut 1      un sujet
  python Evaluation.py ART         les 109 sujets, avec la moyenne finale
  python Evaluation.py ART 4 1     ART au checkpoint de l'epoch 1, sujet 4

"""

import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne
from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

mne.set_log_level("ERROR")

#Chemins des fichiers
Output = pl.Path(r"C:\Users\aymen\Desktop\ART\Output")
Pretraite = Output / "Prétraité"
Nettoye = Output / "Nettoyé"
ART_Epochs = Output / "ART"     # sorties d'un checkpoint LOSO précis (cf. ART_Epochs.py)
classe1, classe2 = "gauche", "droite"   # imagerie main gauche / main droite (runs 4/8/12)

#Nom de fichier selon le signal. Tous viennent de Prétraité, y compris le brut : le repos T0
#est retiré plus bas, donc toutes les méthodes portent sur exactement les mêmes essais.
fichiers = {"brut": None,
            "ART": "ART.fif",
            "ICUNet": "ICUNet.fif",
            "ICUNet++": "ICUNet++.fif",
            "ICUNet_attn": "ICUNet_attn.fif",
            "DuoCL": "DuoCL.fif",
            "GCTNet": "GCTNet.fif",
            "ICLABEL": "ICLABEL.fif"}   # ICA ICLabel sur le raw 64 canaux (ICLABEL_Brut.py)

#Paramètres CSP + LDA (identiques à la référence MNE)
sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))   # fenêtre [1,2]s (+ retard FIR ~1.95s)
N_iter = 10

#Ligne de commande : quel signal évaluer (+ sujet optionnel, sinon tous : 1-109, et epoch
#optionnel pour un checkpoint LOSO précis, comme CSP.py et Visualize.py)
parser = ap.ArgumentParser(description="Évaluation CSP + LDA")
parser.add_argument("Method", help="signal à évaluer (insensible à la casse) : " + ", ".join(fichiers))
parser.add_argument("Sujet", type=int, nargs="?", default=None, help="numéro du sujet (défaut : tous, 1-109)")
parser.add_argument("Epoch", type=int, nargs="?", default=None,
                    help="numéro d'epoch (ART uniquement, ex. 1) -> Output/ART/SXXX/ART_epochN.fif")
args = parser.parse_args()

#Résolution insensible à la casse (ex. "duocl" -> "DuoCL")
correspondance = {nom.lower(): nom for nom in fichiers}
if args.Method.lower() not in correspondance:
    raise SystemExit(f"ERREUR : signal inconnu '{args.Method}'. Choix possibles : {', '.join(fichiers)}")
args.Method = correspondance[args.Method.lower()]

if args.Epoch is not None and (args.Method != "ART" or args.Sujet is None):
    raise SystemExit("ERREUR : l'argument Epoch n'est utilisable qu'avec ART et un sujet précis, "
                     "ex. python Evaluation.py ART 4 1")


#CSP + LDA (régularisation pour ICLabel : données rang-déficientes après retrait de composantes)
reg = "ledoit_wolf" if args.Method == "ICLABEL" else None
clf = Pipeline([("CSP", CSP(n_components=4, reg=reg, log=True, norm_trace=False)),
                ("LDA", LinearDiscriminantAnalysis())])
cv = ShuffleSplit(N_iter, test_size=0.2, random_state=42)

#Évaluation : un seul sujet si précisé, sinon tous (1-109)
sujets = [args.Sujet] if args.Sujet else range(1, 110)
moyennes, ecarts = [], []
for s in sujets:
    sujet_id = "S" + str(s).zfill(3)
    if args.Method == "brut":
        fichier = Pretraite / (sujet_id + "_Pre.fif")
    elif args.Epoch is not None:
        fichier = ART_Epochs / sujet_id / f"ART_epoch{args.Epoch}.fif"
    else:
        fichier = Nettoye / sujet_id / fichiers[args.Method]
    if not fichier.exists():
        print(f"  {args.Method:6s} {sujet_id} : fichier introuvable ({fichier})")
        continue
    epochs = mne.read_epochs(fichier, preload=True)[classe1, classe2]   # retire le repos T0
    y = epochs.events[:, 2]                            # 1=gauche, 2=droite

    #Prétraitement : référence moyenne, band-pass 7-30 Hz, fenêtre [1,2]s
    X = epochs.get_data()
    X = X - X.mean(axis=1, keepdims=True)
    X = filter_data(X, sfreq, fmin, fmax)
    X = X[:, :, crop]

    scores = cross_val_score(clf, X, y, cv=cv)         # N_iter=10 runs (ShuffleSplit)
    moyennes.append(scores.mean())
    ecarts.append(scores.std())
    print(f"  {args.Method:6s} {sujet_id} : moyenne {scores.mean():.2f} +/- {scores.std():.2f}")

if len(moyennes) > 1:
    print(f"\nMoyenne finale ({args.Method}) : {np.mean(moyennes):.3f} +/- {np.std(moyennes):.3f}")
