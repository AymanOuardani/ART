import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne
from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
import Utils

mne.set_log_level("ERROR")

#Chemins des fichiers
Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
Rapport_Tex = pl.Path(r"C:\Users\aymen\Desktop\ART\Résultats\Rapport_ART.tex")

#Libellé de chaque signal dans le tableau du rapport (Résultats/Rapport_ART.pdf)
labels_rapport = {"brut": "Brut",
                  "ART": "ART",
                  "ICUNet": "ICUNet",
                  "ICUNet++": "ICUNet++",
                  "ICUNet_attn": "ICUNet\\_attn",
                  "DuoCL": "DuoCL",
                  "GCTNet": "GCTNet",
                  "ICLABEL": "ICLabel",
                  "ICA": "ICA"}

#Nom de fichier selon le signal (brut = prétraité, sans débruitage)
fichiers = {"brut": None,
            "ART": "ART.fif",
            "ICUNet": "ICUNet.fif",
            "ICUNet++": "ICUNet++.fif",
            "ICUNet_attn": "ICUNet_attn.fif",
            "DuoCL": "DuoCL.fif",
            "GCTNet": "GCTNet.fif",
            "ICLABEL": "ICLABEL.fif",
            "ICA": "ICA.fif"}   # tous (sauf brut) viennent de Prétraité_Total : repos T0 retiré plus bas

#Paramètres CSP + LDA (identiques à la référence MNE)
sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))   # fenêtre [1,2]s (+ retard FIR ~1.95s)
N_iter = 10

#Ligne de commande : quel signal évaluer (+ sujet optionnel, sinon tous : 1-109)
parser = ap.ArgumentParser(description="Évaluation CSP + LDA")
parser.add_argument("Method", help="signal à évaluer (insensible à la casse) : " + ", ".join(fichiers))
parser.add_argument("Sujet", type=int, nargs="?", default=None, help="numéro du sujet (défaut : tous, 1-109)")
args = parser.parse_args()

#Résolution insensible à la casse (ex. "duocl" -> "DuoCL")
correspondance = {nom.lower(): nom for nom in fichiers}
if args.Method.lower() not in correspondance:
    raise SystemExit(f"ERREUR : signal inconnu '{args.Method}'. Choix possibles : {', '.join(fichiers)}")
args.Method = correspondance[args.Method.lower()]


def prep(X):
    X = X - X.mean(axis=1, keepdims=True)              # référence moyenne
    X = filter_data(X, sfreq, fmin, fmax)              # band-pass 7-30 Hz
    return X[:, :, crop]                                # fenêtre [1,2]s


#CSP + LDA (régularisation pour ICA/ICLabel : données rang-déficientes après retrait de composantes)
reg = "ledoit_wolf" if args.Method in ("ICLABEL", "ICA") else None
clf = Pipeline([("CSP", CSP(n_components=6, reg=reg, log=True, norm_trace=False)),
                ("LDA", LinearDiscriminantAnalysis())])
cv = ShuffleSplit(N_iter, test_size=0.2, random_state=42)

#Évaluation : un seul sujet si précisé, sinon tous (1-109)
sujets = [args.Sujet] if args.Sujet else range(1, 110)
moyennes, ecarts = [], []
for s in sujets:
    sujet_id = "S" + str(s).zfill(3)
    if args.Method == "brut":
        fichier = Pretraite / (sujet_id + "-epo.fif")
    else:
        fichier = Nettoye / sujet_id / fichiers[args.Method]
    if not fichier.exists():
        print(f"  {args.Method:6s} {sujet_id} : fichier introuvable ({fichier})")
        continue
    epochs = mne.read_epochs(fichier, preload=True)
    if args.Method != "brut":
        epochs = epochs["gauche", "droite"]           # retire le repos (tous générés depuis Prétraité_Total)
    X = epochs.get_data()
    y = epochs.events[:, 2]                            # 1=gauche, 2=droite
    scores = cross_val_score(clf, prep(X), y, cv=cv)   # N_iter=10 runs (ShuffleSplit)
    moyennes.append(scores.mean())
    ecarts.append(scores.std())
    print(f"  {args.Method:6s} {sujet_id} : moyenne {scores.mean():.2f} +/- {scores.std():.2f}")

if len(moyennes) > 1:
    print(f"\nMoyenne finale ({args.Method}) : {np.mean(moyennes):.3f} +/- {np.std(moyennes):.3f}")

#Sujet unique évalué avec succès -> reporte la valeur dans Résultats/Rapport_ART.pdf
if args.Sujet and moyennes and Rapport_Tex.exists():
    sujet_id = "S" + str(args.Sujet).zfill(3)
    ligne = f"{labels_rapport[args.Method]} & {moyennes[0]:.2f} $\\pm$ {ecarts[0]:.2f} \\\\"
    Utils.maj_tableau_tex(Rapport_Tex.parent / "data" / f"{sujet_id}_accuracy.json",
                          Rapport_Tex.parent / "data" / f"{sujet_id}_accuracy.tex",
                          args.Method, ligne, macro="accuracyrows", ordre=list(fichiers))
    Utils.recompile_latex(Rapport_Tex)
