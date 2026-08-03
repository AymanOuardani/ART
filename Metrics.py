"""
Métriques d'un sujet, méthode par méthode, face à la référence ICLabel.

  python Metrics.py RMS  1          RMS du signal de chaque méthode, en µV
  python Metrics.py RMSE 1          écart quadratique moyen à ICLabel, en µV
  python Metrics.py MAE  1          écart absolu moyen à ICLabel, en µV
  python Metrics.py SNR  1          20·log10(RMS ICLabel / RMSE), en dB
  python Metrics.py tout 1          les quatre d'un coup

  python Metrics.py tout 1 --epoch 40    ART pris dans Output/ART/S001/ART_epoch40.fif
                                         (checkpoint LOSO précis) au lieu du modèle d'origine

Le SNR se lit comme un rapport signal/bruit : 0 dB signifie que l'erreur a la même
amplitude que le signal de référence, 6 dB qu'elle en fait la moitié, 14 dB le cinquième.
"""

import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne

mne.set_log_level("ERROR")

#Chemins des fichiers
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
ART_Epochs = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\ART")

#ICLabel est la référence : l'ICA décomposée sur le signal continu à 64 canaux (ICLABEL_Brut.py)
Reference = "ICLABEL.fif"
methodes = {"ART": "ART.fif", "ICUNet": "ICUNet.fif", "ICUNet++": "ICUNet++.fif",
            "ICUNet_attn": "ICUNet_attn.fif", "DuoCL": "DuoCL.fif", "GCTNet": "GCTNet.fif"}

METRIQUES = ["RMS", "RMSE", "MAE", "SNR", "tout"]

#Ligne de commande : quelle métrique, quel sujet
parser = ap.ArgumentParser(description="Métriques d'un sujet face à ICLabel",
                           formatter_class=ap.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument("Metrique", help="métrique à afficher (insensible à la casse) : " + ", ".join(METRIQUES))
parser.add_argument("Sujet", type=int, help="numéro du sujet (1-109)")
parser.add_argument("--epoch", type=int, default=None,
                    help="numéro d'epoch : prend ART dans Output/ART/SXXX/ART_epochN.fif")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

correspondance = {m.lower(): m for m in METRIQUES}
if args.Metrique.lower() not in correspondance:
    raise SystemExit(f"ERREUR : métrique inconnue '{args.Metrique}'. Choix : {', '.join(METRIQUES)}")
args.Metrique = correspondance[args.Metrique.lower()]

#Lecture de la référence
f_ref = Nettoye / sujet_id / Reference
if not f_ref.exists():
    raise SystemExit(f"ERREUR : référence introuvable : {f_ref}")
iclabel = mne.read_epochs(f_ref, preload=True)
ref = iclabel.get_data()
rms_ref = np.sqrt(np.mean(ref ** 2)) * 1e6


#Les quatre métriques de chaque méthode face à la référence
resultats = {}
for nom, fichier in methodes.items():

    #Chemin du fichier (ART avec --epoch : signal d'un checkpoint LOSO précis)
    if nom == "ART" and args.epoch is not None:
        chemin = ART_Epochs / sujet_id / f"ART_epoch{args.epoch}.fif"
        nom = f"ART (epoch {args.epoch})"
    else:
        chemin = Nettoye / sujet_id / fichier
    if not chemin.exists():
        resultats[nom] = "fichier introuvable"
        continue

    #La comparaison se fait échantillon par échantillon : les essais doivent correspondre
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

#Affichage : la ou les colonnes demandées
colonnes = ["RMS", "RMSE", "MAE", "SNR"] if args.Metrique == "tout" else [args.Metrique]
unites = {"RMS": "µV", "RMSE": "µV", "MAE": "µV", "SNR": "dB"}

print(f"\n{sujet_id} — référence ICLabel : RMS {rms_ref:.2f} µV")
print("  " + "méthode".ljust(18) + "".join(f"{c + ' (' + unites[c] + ')':>14}" for c in colonnes))
for nom, m in resultats.items():
    if isinstance(m, str):
        print("  " + nom.ljust(18) + f"  {m}")
    else:
        print("  " + nom.ljust(18) + "".join(f"{m[c]:>14.2f}" for c in colonnes))
