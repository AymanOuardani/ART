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
Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")

#Nom de fichier selon le signal (brut = prétraité, sans débruitage)
fichiers = {"brut": None,
            "ART": "ART.fif",
            "ICUNet": "ICUNet.fif",
            "ICUNet++": "ICUNet++.fif",
            "ICUNet_attn": "ICUNet_attn.fif",
            "DuoCL": "DuoCL.fif",
            "GCTNet": "GCTNet.fif",
            "ICLABEL": "ICLABEL.fif"}   # ICLABEL garde le repos T0 : retiré plus bas

#Paramètres CSP + LDA (identiques à la référence MNE)
sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))   # fenêtre [1,2]s (+ retard FIR ~1.95s)
N_iter = 10

#Ligne de commande : quel signal évaluer
parser = ap.ArgumentParser(description="Évaluation CSP + LDA")
parser.add_argument("Method", choices=list(fichiers), help="signal à évaluer")
args = parser.parse_args()


def prep(X):
    X = X - X.mean(axis=1, keepdims=True)              # référence moyenne
    X = filter_data(X, sfreq, fmin, fmax)              # band-pass 7-30 Hz
    return X[:, :, crop]                                # fenêtre [1,2]s


#CSP + LDA (régularisation pour ICLabel : données rang-déficientes après retrait de composantes ICA)
reg = "ledoit_wolf" if args.Method == "ICLABEL" else None
clf = Pipeline([("CSP", CSP(n_components=4, reg=reg, log=True, norm_trace=False)),
                ("LDA", LinearDiscriminantAnalysis())])
cv = ShuffleSplit(N_iter, test_size=0.2, random_state=42)

#Évaluation sujet par sujet
moyennes = []
for s in range(1, 110):
    sujet_id = "S" + str(s).zfill(3)
    if args.Method == "brut":
        fichier = Pretraite / (sujet_id + "-epo.fif")
    else:
        fichier = Nettoye / sujet_id / fichiers[args.Method]
    if not fichier.exists():
        continue
    epochs = mne.read_epochs(fichier, preload=True)
    if args.Method == "ICLABEL":
        epochs = epochs["gauche", "droite"]           # retire le repos (éval 2 classes)
    X = epochs.get_data()
    y = epochs.events[:, 2]                            # 1=gauche, 2=droite
    scores = cross_val_score(clf, prep(X), y, cv=cv)
    moyennes.append(scores.mean())
    print(f"  {args.Method:6s} {sujet_id} : moyenne {scores.mean():.2f}")

print(f"\nMoyenne finale ({args.Method}) : {np.mean(moyennes):.3f} +/- {np.std(moyennes):.3f}")
