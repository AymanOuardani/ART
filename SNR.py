import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne

mne.set_log_level("ERROR")

#Chemins des fichiers
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")

#Modèles comparés à ICLabel (référence)
modeles = ["ART", "ICUNet", "ICUNet++", "ICUNet_attn", "DuoCL", "GCTNet"]

#Ligne de commande : numéro du sujet
parser = ap.ArgumentParser(description="SNR (dB) par rapport à ICLabel, pour un sujet, par modèle")
parser.add_argument("Sujet", type=int, help="Numéro du sujet (1-109)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

#Lecture d'ICLabel (référence)
f_iclabel = Nettoye / sujet_id / "ICLABEL.fif"
if not f_iclabel.exists():
    raise SystemExit(f"ERREUR : fichier introuvable : {f_iclabel}")
iclabel = mne.read_epochs(f_iclabel, preload=True)
rms_iclabel = np.sqrt(np.mean(iclabel.get_data() ** 2))

#Calcul du SNR, modèle par modèle : 20*log10(RMS_ICLABEL / RMSE)
for modele in modeles:
    fichier = Nettoye / sujet_id / (modele + ".fif")
    if not fichier.exists():
        print(f"SNR de {modele} = fichier introuvable")
        continue
    epochs = mne.read_epochs(fichier, preload=True)
    if not np.array_equal(epochs.events[:, 2], iclabel.events[:, 2]):
        print(f"SNR de {modele} = essais non alignés avec ICLabel")
        continue
    mse = np.mean((epochs.get_data() - iclabel.get_data()) ** 2)
    rmse = np.sqrt(mse)
    snr = 20 * np.log10(rms_iclabel / rmse)
    print(f"SNR de {modele} = {snr:.2f} dB")
