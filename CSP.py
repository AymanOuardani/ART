"""
Montre ce que le décodeur regarde : les topographies des filtres CSP ajustés sur un signal,
c'est-à-dire les zones du scalp qui séparent le mieux main gauche et main droite. Un bon
débruitage doit laisser apparaître les aires sensori-motrices, de part et d'autre du vertex.

  python CSP.py brut 1        signal brut du sujet 1
  python CSP.py ICLABEL 1
  python CSP.py ART 4 1       ART au checkpoint LOSO de l'epoch 1

La figure compare trois lignes : le signal demandé, ICLabel (la cible) et ART d'origine.
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
ART_Epochs = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\ART")

#Nom de fichier selon le signal (brut = prétraité, sans débruitage), comme Evaluation.py
fichiers = {"brut": None,
            "ART": "ART.fif",
            "ICUNet": "ICUNet.fif",
            "ICUNet++": "ICUNet++.fif",
            "ICUNet_attn": "ICUNet_attn.fif",
            "DuoCL": "DuoCL.fif",
            "GCTNet": "GCTNet.fif",
            "ICLABEL": "ICLABEL.fif"}   # ICLabel = ICA sur le raw 64 canaux

#Paramètres CSP identiques à Evaluation.py (imagerie main gauche vs main droite, runs 4/8/12)
sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))   # fenêtre [1,2]s (+ retard FIR ~1.95s)
n_components = 4

#Ligne de commande : quel signal, quel sujet (+ epoch optionnel pour ART)
parser = ap.ArgumentParser(description="Composantes CSP d'un sujet")
parser.add_argument("Signal", help="signal à décomposer (insensible à la casse) : " + ", ".join(fichiers))
parser.add_argument("Sujet", type=int, help="numéro du sujet (1-109)")
parser.add_argument("Epoch", type=int, nargs="?", default=None,
                    help="numéro d'epoch (ART uniquement, ex. 60) -> Output/ART/SXXX/ART_epochN.fif")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

#Résolution insensible à la casse (ex. "iclabel" -> "ICLABEL")
correspondance = {nom.lower(): nom for nom in fichiers}
if args.Signal.lower() not in correspondance:
    raise SystemExit(f"ERREUR : signal inconnu '{args.Signal}'. Choix possibles : {', '.join(fichiers)}")
args.Signal = correspondance[args.Signal.lower()]

if args.Epoch is not None and args.Signal != "ART":
    raise SystemExit("ERREUR : l'argument Epoch n'est utilisable qu'avec le signal ART.")

#Les trois signaux tracés : celui demandé, ICLabel (la cible) et ART d'origine (non réentraîné)
a_tracer = [(args.Signal, args.Epoch), ("ICLABEL", None), ("ART", None)]

#Décomposition CSP de chacun
noms, patterns, info = [], [], None
for signal, epoch_num in a_tracer:

    #Chemin du fichier (ART avec un numéro d'epoch : checkpoint LOSO)
    if signal == "brut":
        fichier = Pretraite / (sujet_id + "_Pre.fif")
    elif epoch_num is not None:
        fichier = ART_Epochs / sujet_id / f"ART_epoch{epoch_num}.fif"
    else:
        fichier = Nettoye / sujet_id / fichiers[signal]
    if not fichier.exists():
        raise SystemExit(f"ERREUR : fichier introuvable : {fichier}")

    #Lecture des essais des deux classes de mouvement (le repos T0 est retiré)
    epochs = mne.read_epochs(fichier, preload=True)["gauche", "droite"]
    if info is None:
        info = epochs.info

    #Prétraitement : référence moyenne, band-pass 7-30 Hz, fenêtre [1,2]s
    X = epochs.get_data()
    X = X - X.mean(axis=1, keepdims=True)
    X = filter_data(X, sfreq, fmin, fmax)
    X = X[:, :, crop]

    #Ajustement de la CSP (régularisation pour ICLabel : données rang-déficientes)
    reg = "ledoit_wolf" if signal == "ICLABEL" else None
    csp = CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)
    csp.fit(X, epochs.events[:, 2])

    nom = "ART_Orig" if (signal == "ART" and epoch_num is None) else signal
    noms.append(nom + (f" (epoch {epoch_num})" if epoch_num else ""))
    patterns.append(csp.patterns_[:n_components])

#Topographies des composantes : ce que chaque filtre CSP lit à la surface du scalp
fig, axes = plt.subplots(len(noms), n_components, figsize=(3 * n_components, 11))
for ligne in range(len(noms)):
    for i in range(n_components):
        mne.viz.plot_topomap(patterns[ligne][i], info, axes=axes[ligne, i], show=False)
        axes[ligne, i].set_title(noms[ligne] + f" - CSP {i + 1}", fontsize=10)

fig.suptitle(sujet_id)
fig.tight_layout(h_pad=3)
plt.show(block=True)
