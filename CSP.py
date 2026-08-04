"""
Montre les topographies des composantes du CSP.

  python CSP.py brut 1   
  python CSP.py ICLABEL 1
  python CSP.py ART_Local 1

La figure compare trois lignes : le signal demandé, ICLabel et ART_Orig.
"""

import argparse as ap
import pathlib as pl
import matplotlib.pyplot as plt
import mne as mne
from mne.filter import filter_data
from mne.decoding import CSP

mne.set_log_level("ERROR")

#Chemins des fichiers
Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")

#Fichier de chaque signal
fichiers = {"brut": None,
            "ART_Orig": "ART_Orig.fif",
            "ART_Local": "ART_Local.fif",
            "ICUNet": "ICUNet.fif",
            "ICUNet++": "ICUNet++.fif",
            "ICUNet_attn": "ICUNet_attn.fif",
            "DuoCL": "DuoCL.fif",
            "GCTNet": "GCTNet.fif",
            "ICLABEL": "ICLABEL.fif"}   # ICLabel = ICA sur le raw 64 canaux

#Paramètres CSP, identiques à Evaluation.py
sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))   # fenêtre [1,2]s (+ retard FIR ~1.95s)
n_components = 4

#Ligne de commande
parser = ap.ArgumentParser(description="Composantes CSP d'un sujet")
parser.add_argument("Signal", help="signal à décomposer (insensible à la casse) : " + ", ".join(fichiers))
parser.add_argument("Sujet", type=int, help="numéro du sujet (1-109)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

#Insensible à la casse
correspondance = {nom.lower(): nom for nom in fichiers}
if args.Signal.lower() not in correspondance:
    raise SystemExit(f"ERREUR : signal inconnu '{args.Signal}'. Choix possibles : {', '.join(fichiers)}")
args.Signal = correspondance[args.Signal.lower()]

#Trois signaux tracés : le demandé, ICLabel et ART_Orig
a_tracer = [args.Signal, "ICLABEL", "ART_Orig"]

#Décomposition CSP de chacun
noms, patterns, info = [], [], None
for signal in a_tracer:

    #Chemin du fichier
    if signal == "brut":
        fichier = Pretraite / (sujet_id + "_Pre.fif")
    else:
        fichier = Nettoye / sujet_id / fichiers[signal]
    if not fichier.exists():
        raise SystemExit(f"ERREUR : fichier introuvable : {fichier}")

    #Essais des deux classes, repos retiré
    epochs = mne.read_epochs(fichier, preload=True)["gauche", "droite"]
    if info is None:
        info = epochs.info

    #Référence moyenne, 7-30 Hz, fenêtre [1,2]s
    X = epochs.get_data()
    X = X - X.mean(axis=1, keepdims=True)
    X = filter_data(X, sfreq, fmin, fmax)
    X = X[:, :, crop]

    #ICLabel est rang-déficient, d'où la régularisation
    reg = "ledoit_wolf" if signal == "ICLABEL" else None
    csp = CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)
    csp.fit(X, epochs.events[:, 2])

    noms.append(signal)
    patterns.append(csp.patterns_[:n_components])

#Topographies : ce que chaque filtre lit sur le scalp
fig, axes = plt.subplots(len(noms), n_components, figsize=(3 * n_components, 11))
for ligne in range(len(noms)):
    for i in range(n_components):
        mne.viz.plot_topomap(patterns[ligne][i], info, axes=axes[ligne, i], show=False)
        axes[ligne, i].set_title(noms[ligne] + f" - CSP {i + 1}", fontsize=10)

fig.suptitle(sujet_id)
fig.tight_layout(h_pad=3)
plt.show(block=True)
