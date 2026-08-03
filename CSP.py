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


def prep(X):
    X = X - X.mean(axis=1, keepdims=True)              # référence moyenne
    X = filter_data(X, sfreq, fmin, fmax)              # band-pass 7-30 Hz
    return X[:, :, crop]                                # fenêtre [1,2]s


def lit_epochs(signal, epoch_num):
    # Epochs des deux classes de mouvement (ART avec un numéro d'epoch : checkpoint LOSO)
    if signal == "brut":
        fichier = Pretraite / (sujet_id + "_Pre.fif")
    elif epoch_num is not None:
        fichier = ART_Epochs / sujet_id / f"ART_epoch{epoch_num}.fif"
    else:
        fichier = Nettoye / sujet_id / fichiers[signal]
    if not fichier.exists():
        raise SystemExit(f"ERREUR : fichier introuvable : {fichier}")
    return mne.read_epochs(fichier, preload=True)["gauche", "droite"]   # retire le repos T0


def patterns(epochs, signal):
    # Ajustement de la CSP (régularisation pour ICA/ICLabel : données rang-déficientes)
    reg = "ledoit_wolf" if signal == "ICLABEL" else None
    csp = CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)
    csp.fit(prep(epochs.get_data()), epochs.events[:, 2])
    return csp.patterns_[:n_components]


#Ligne 1 : le signal demandé, ligne 2 : ICLabel (cible), ligne 3 : ART d'origine (non réentraîné)
epochs = lit_epochs(args.Signal, args.Epoch)
lignes = [(args.Signal + (f" (epoch {args.Epoch})" if args.Epoch else ""), patterns(epochs, args.Signal)),
          ("ICLabel", patterns(lit_epochs("ICLABEL", None), "ICLABEL")),
          ("ART_Orig", patterns(lit_epochs("ART", None), "ART"))]

#Topographies des composantes : ce que chaque filtre CSP lit à la surface du scalp
fig, axes = plt.subplots(len(lignes), n_components, figsize=(3 * n_components, 11))
for ligne, (nom, pats) in enumerate(lignes):
    for i in range(n_components):
        mne.viz.plot_topomap(pats[i], epochs.info, axes=axes[ligne, i], show=False)
        axes[ligne, i].set_title(nom + f" - CSP {i + 1}", fontsize=10)

fig.suptitle(sujet_id)
fig.tight_layout(h_pad=3)
plt.show(block=True)