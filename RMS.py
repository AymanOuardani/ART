import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne

mne.set_log_level("ERROR")

#Chemins des fichiers
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")

#Modèles à comparer (fichier SXXX/{modele}.fif dans Nettoyé/)
modeles = ["ICLABEL", "ART", "ICUNet", "ICUNet++", "ICUNet_attn", "DuoCL", "GCTNet"]

#Ligne de commande : numéro du sujet
parser = ap.ArgumentParser(description="RMS (µV) du signal nettoyé pour un sujet, par modèle")
parser.add_argument("Sujet", type=int, help="Numéro du sujet (1-109)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

#Calcul du RMS, modèle par modèle
for modele in modeles:
    fichier = Nettoye / sujet_id / (modele + ".fif")
    if not fichier.exists():
        print(f"RMS de {modele} = fichier introuvable")
        continue
    data = mne.read_epochs(fichier, preload=True).get_data()
    rms = np.sqrt(np.mean(data ** 2)) * 1e6   # V -> µV
    print(f"RMS de {modele} = {rms:.2f} µV")
