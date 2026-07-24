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
            "ART": "ART_original-epo.fif",
            "glue": "ART_glue-epo.fif",
            "ICUNet": "ICUNet-epo.fif"}

#Paramètres CSP + LDA (identiques à la référence MNE)
sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))   # fenêtre [1,2]s (+ retard FIR ~1.95s)
N_iter = 10

#Ligne de commande : quel signal évaluer
parser = ap.ArgumentParser(description="Évaluation CSP + LDA")
parser.add_argument("Signal", choices=list(fichiers), help="signal à évaluer")
args = parser.parse_args()


def prep(X):
    X = X - X.mean(axis=1, keepdims=True)              # référence moyenne
    X = filter_data(X, sfreq, fmin, fmax)              # band-pass 7-30 Hz
    return X[:, :, crop]                                # fenêtre [1,2]s


#CSP + LDA
clf = Pipeline([("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                ("LDA", LinearDiscriminantAnalysis())])
cv = ShuffleSplit(N_iter, test_size=0.2, random_state=42)

#Évaluation sujet par sujet
moyennes = []
for s in range(1, 110):
    sujet_id = "S" + str(s).zfill(3)
    if fichiers[args.Signal] is None:
        fichier = Pretraite / (sujet_id + "-epo.fif")
    else:
        fichier = Nettoye / sujet_id / fichiers[args.Signal]
    if not fichier.exists():
        continue
    epochs = mne.read_epochs(fichier, preload=True)
    X = epochs.get_data()
    y = epochs.events[:, 2]                            # 1=gauche, 2=droite
    scores = cross_val_score(clf, prep(X), y, cv=cv)
    moyennes.append(scores.mean())
    print(f"  {args.Signal:6s} {sujet_id} : moyenne {scores.mean():.2f}")

print(f"\nMoyenne finale ({args.Signal}) : {np.mean(moyennes):.3f} +/- {np.std(moyennes):.3f}")
