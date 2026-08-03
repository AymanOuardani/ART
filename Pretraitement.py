import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne
from fractions import Fraction
from scipy.signal import resample_poly, firwin, lfilter
from mne.datasets import eegbci

mne.set_log_level("ERROR")

#Chemins des fichiers : imagerie main gauche / main droite (runs 4/8/12) uniquement
Brut_fif = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI")
Brut_fif_ICLABEL = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI_ICLABEL")
Output = pl.Path(r"C:\Users\aymen\Desktop\ART\Output")
Pretraite = Output / "Prétraité"
Nettoye = Output / "Nettoyé"

epoch_len = 1024           # 4 s à 256 Hz
classe1, classe2 = "gauche", "droite"

#30 canaux du template ART, dans l'ordre attendu par le modèle
Art_Template = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz",
                "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8", "TP7", "CP3", "CPz",
                "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]

#Ligne de commande : numéro du sujet, et quelle version du signal continu prétraiter
parser = ap.ArgumentParser(description="Prétraitement EEGBCI")
parser.add_argument("Sujet", type=int, help="Numéro du sujet (1-109)")
parser.add_argument("--iclabel", action="store_true",
                    help="prétraite le signal nettoyé par ICLabel (cf. ICLABEL_Brut.py) au lieu du "
                         "brut -> Output/Nettoyé/SXXX/ICLABEL_Amélioré.fif")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

#Le brut et sa version nettoyée par ICLabel passent par ce même script, donc par exactement le
#même réordonnancement de canaux, le même rééchantillonnage, le même filtre et le même découpage.
#C'est indispensable : les deux signaux sont ensuite comparés échantillon par échantillon
#(RMSE, SNR) et servent de paire d'entraînement à ART.
source = (Brut_fif_ICLABEL if args.iclabel else Brut_fif) / (sujet_id + "-raw.fif")
if not source.exists():
    raise SystemExit(f"ERREUR : fichier introuvable : {source}"
                     + ("\n  (lance d'abord ICLABEL_Brut.py)" if args.iclabel else ""))

#Lecture du fichier brut regroupé et standardisation des noms de canaux
raw = mne.io.read_raw_fif(source, preload=True)
eegbci.standardize(raw)

#Sélection et réordonnancement des 30 canaux ART (par nom)
idx = {c.upper(): i for i, c in enumerate(raw.ch_names)}
order = [idx[nom.upper()] for nom in Art_Template]
data = raw.get_data()[order, :] * 1e6                          # (30, T) en microvolts

#Rééchantillonnage 160 -> 256 Hz
frac = Fraction(256, int(raw.info["sfreq"])).limit_denominator()
data = resample_poly(data, frac.numerator, frac.denominator, axis=1)

#Filtre band-pass FIR 1-50 Hz (1000 taps)
coeff = firwin(1000, [1, 50], pass_zero=False, fs=256.0)
data = lfilter(coeff, 1.0, data, axis=1)

#Tout le signal : les deux classes de mouvement et le repos (T0), jamais découpé autrement
events, eid = mne.events_from_annotations(raw)
scale = 256 / raw.info["sfreq"]                                # indices 160 Hz -> 256 Hz
labels = {eid["T0"]: 3, eid["T1"]: 1, eid["T2"]: 2}           # repos=3, classe1=1, classe2=2
event_id = {classe1: 1, classe2: 2, "repos": 3}
out = (Nettoye / sujet_id / "ICLABEL_Amélioré.fif") if args.iclabel else (Pretraite / (sujet_id + "_Pre.fif"))

#Découpage en blocs de 4 s autour des événements retenus
X, y = [], []
for onset, _, code in events:
    if code not in labels:
        continue
    start = int(round(onset * scale))
    seg = data[:, start:start + epoch_len]
    if seg.shape[1] == epoch_len:
        X.append(seg)
        y.append(labels[code])
X = np.array(X)
y = np.array(y)

#Construction de l'objet Epochs
info = mne.create_info(Art_Template, sfreq=256, ch_types="eeg")
info.set_montage("standard_1005", on_missing="ignore")
ev = np.column_stack([np.arange(len(y)), np.zeros(len(y), int), y])
epochs = mne.EpochsArray(X * 1e-6, info, ev, tmin=0, event_id=event_id)

#Sauvegarde
out.parent.mkdir(parents=True, exist_ok=True)
epochs.save(out, overwrite=True)
print("Prétraitement sauvegardé pour Sujet ", args.Sujet, ":", len(y), "blocs ->", out)