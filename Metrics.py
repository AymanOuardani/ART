"""
Métriques d'un sujet, méthode par méthode.

  python Metrics.py RMS  1          RMS du signal de chaque méthode, en µV
  python Metrics.py RMSE 1          écart quadratique moyen à ICLabel, en µV
  python Metrics.py MAE  1          écart absolu moyen à ICLabel, en µV
  python Metrics.py SNR  1          20·log10(RMS ICLabel / RMSE), en dB
  python Metrics.py tout 1          les quatre d'un coup

Le SNR se lit comme un rapport signal/bruit : 0 dB signifie que l'erreur a la même
amplitude que le signal de référence, 6 dB qu'elle en fait la moitié, 14 dB le cinquième.
"""

"""
English summary: computes and prints signal-quality metrics (RMS, RMSE, MAE, SNR) for
every denoising method applied to one subject, all measured against the ICLabel signal
used as ground-truth reference.

Usage:
  python Metrics.py <metric> <subject>
  <metric> is one of: RMS, RMSE, MAE, SNR, tout (all four). <subject> is 1-109.
"""

import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne

mne.set_log_level("ERROR")

#Chemins des fichiers
Racine = pl.Path(__file__).resolve().parent
Nettoye = Racine / "Output" / "Nettoyé"

#ICLabel sert de référence
Reference = "ICLABEL.fif"
methodes = {"ART_Orig": "ART_Orig.fif", "ART_Local": "ART_Local.fif",
            "ICUNet": "ICUNet.fif", "ICUNet++": "ICUNet++.fif",
            "ICUNet_attn": "ICUNet_attn.fif", "DuoCL": "DuoCL.fif", "GCTNet": "GCTNet.fif"}

Metriques = ["RMS", "RMSE", "MAE", "SNR", "tout"]

#Ligne de commande
parser = ap.ArgumentParser(description="Métriques d'un sujet face à ICLabel",
                           formatter_class=ap.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument("Metrique", help="métrique à afficher (insensible à la casse) : " + ", ".join(Metriques))
parser.add_argument("Sujet", type=int, help="numéro du sujet (1-109)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

correspondance = {m.lower(): m for m in Metriques}
if args.Metrique.lower() not in correspondance:
    raise SystemExit(f"ERREUR : métrique inconnue '{args.Metrique}'. Choix : {', '.join(Metriques)}")
args.Metrique = correspondance[args.Metrique.lower()]

#Lecture de la référence
f_ref = Nettoye / sujet_id / Reference
if not f_ref.exists():
    raise SystemExit(f"ERREUR : référence introuvable : {f_ref}")
iclabel = mne.read_epochs(f_ref, preload=True)
ref = iclabel.get_data()
rms_ref = np.sqrt(np.mean(ref ** 2)) * 1e6


#Métriques de chaque méthode
resultats = {}
for nom, fichier in methodes.items():

    chemin = Nettoye / sujet_id / fichier
    if not chemin.exists():
        resultats[nom] = "fichier introuvable"
        continue

    #Les essais doivent correspondre à ceux d'ICLabel
    epochs = mne.read_epochs(chemin, preload=True)
    if not np.array_equal(epochs.events[:, 2], iclabel.events[:, 2]):
        resultats[nom] = "essais non alignés avec ICLabel"
        continue

    data = epochs.get_data()
    rmse = np.sqrt(np.mean((data - ref) ** 2)) * 1e6
    resultats[nom] = {"RMS": np.sqrt(np.mean(data ** 2)) * 1e6,
                      "RMSE": rmse,
                      "MAE": np.mean(np.abs(data - ref)) * 1e6,
                      "SNR": 20 * np.log10(rms_ref / rmse)}

#Affichage
colonnes = ["RMS", "RMSE", "MAE", "SNR"] if args.Metrique == "tout" else [args.Metrique]
unites = {"RMS": "µV", "RMSE": "µV", "MAE": "µV", "SNR": "dB"}

print(f"\n{sujet_id} — référence ICLabel : RMS {rms_ref:.2f} µV")
print("  " + "méthode".ljust(18) + "".join(f"{c + ' (' + unites[c] + ')':>14}" for c in colonnes))
for nom, m in resultats.items():
    if isinstance(m, str):
        print("  " + nom.ljust(18) + f"  {m}")
    else:
        print("  " + nom.ljust(18) + "".join(f"{m[c]:>14.2f}" for c in colonnes))
