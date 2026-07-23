"""
Etape 1/3 — Pretraitement EEGBCI -> format ART (art. 2.3), motor imagery gauche/droite.
  - 64 -> 30 canaux : les canaux du template ART, selectionnes par nom
  - 160 -> 256 Hz   : resample rationnel 8/5
  - band-pass 1-50 Hz : FIR 1000 taps (identique a main.py)
Puis decoupe en essais de 4 s (1024 pts) autour de T1 (gauche) / T2 (droite).

    python Pretreatement.py 1-109            # un fichier par sujet
    python Pretreatement.py 1                # test rapide (defaut)

Sortie : Output/Prétraité/S001-epo.fif, S002-epo.fif, ... (objets MNE Epochs).
Etapes suivantes : clean.py (debruitage) puis Evaluation.py (evaluation).
"""

import argparse
from pathlib import Path
from fractions import Fraction

import numpy as np
import pandas as pd
import mne
from scipy.signal import resample_poly, firwin, lfilter
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

mne.set_log_level("ERROR")

# ------------------------------- Config -----------------------------------
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "Databases" / "EEGBCI" / "Brut"          # EDF bruts telecharges ici
PRETRAITE = BASE / "Output" / "Prétraité"   # sorties pretraitees ici

RUNS = [4, 8, 12]          # motor imagery main gauche (T1) / droite (T2)
EPOCH_LEN = 1024           # 4 s a 256 Hz

# 30 canaux du template ART, dans l'ordre attendu par le modele
ART_TEMPLATE = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz",
                "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8", "TP7", "CP3", "CPz",
                "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]


def resample(signal, fs, tgt_fs):
    # Resample rationnel (ex. 160->256 = 8/5), canal par canal
    frac = Fraction(int(tgt_fs), int(fs)).limit_denominator()
    return resample_poly(signal, frac.numerator, frac.denominator, axis=1)


def fir_filter(signal, lowcut, highcut, fs=256.0):
    # Band-pass FIR 1000 taps, identique a main.py
    coeff = firwin(1000, [lowcut, highcut], pass_zero=False, fs=fs)
    return lfilter(coeff, 1.0, signal, axis=1)


def load_raw(subject):
    # Telecharge (dans Databases/EEGBCI/Brut) et concatene les runs 4/8/12 d'un sujet
    fnames = eegbci.load_data(subjects=[subject], runs=RUNS, path=str(RAW_DIR),
                              update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames],
                           verbose="ERROR")
    eegbci.standardize(raw)
    return raw


def pick_30(raw):
    # Reordonne les 64 canaux vers les 30 du template ART (par nom)
    idx = {c.upper(): i for i, c in enumerate(raw.ch_names)}
    order = [idx[name.upper()] for name in ART_TEMPLATE]
    return raw.get_data()[order, :]                     # (30, T) en volts


def preprocess_subject(subject):
    # Renvoie X (n_essais, 30, 1024) et y (0=gauche, 1=droite) pour un sujet
    raw = load_raw(subject)
    data = pick_30(raw) * 1e6                           # -> microvolts
    data = fir_filter(resample(data, raw.info["sfreq"], 256), 1, 50)

    events, eid = mne.events_from_annotations(raw, verbose="ERROR")
    scale = 256 / raw.info["sfreq"]                     # indices 160 Hz -> 256 Hz
    labels = {eid["T1"]: 0, eid["T2"]: 1}

    X, y = [], []
    for onset, _, code in events:
        if code not in labels:
            continue
        start = int(round(onset * scale))
        seg = data[:, start:start + EPOCH_LEN]
        if seg.shape[1] == EPOCH_LEN:
            X.append(seg)
            y.append(labels[code])
    return np.array(X), np.array(y)


def parse_subjects(text):
    # "1-109" ou "1,2,5" -> liste d'entiers
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def to_epochs(X, y, subjects):
    # Construit un objet MNE Epochs (30 canaux, 256 Hz, labels gauche/droite)
    info = mne.create_info(ART_TEMPLATE, sfreq=256, ch_types="eeg")
    info.set_montage("standard_1005", on_missing="ignore")
    events = np.column_stack([np.arange(len(y)), np.zeros(len(y), int), y + 1])
    return mne.EpochsArray(X * 1e-6, info, events, tmin=0,
                           event_id={"gauche": 1, "droite": 2},
                           metadata=pd.DataFrame({"subject": subjects}),
                           verbose="ERROR")


def save_subject(X, y, subject):
    # Ecrit un fichier par sujet dans Output/Prétraité (S001-epo.fif)
    out = PRETRAITE / f"S{subject:03d}-epo.fif"
    to_epochs(X, y, np.full(len(y), subject)).save(out, overwrite=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Etape 1/3 : pretraitement EEGBCI -> format ART.")
    ap.add_argument("subjects", nargs="?", default="1", help="ex. 1-109 ou 1,2,5 (defaut : 1)")
    args = ap.parse_args()

    PRETRAITE.mkdir(parents=True, exist_ok=True)
    tot_l = tot_r = 0
    for s in parse_subjects(args.subjects):
        Xs, ys = preprocess_subject(s)
        out = save_subject(Xs, ys, s)
        tot_l += (ys == 0).sum()
        tot_r += (ys == 1).sum()
        print(f"  sujet {s:>3d} : {(ys == 0).sum()} gauche / {(ys == 1).sum()} droite -> {out.name}")

    print(f"\nTotal : {tot_l} gauche / {tot_r} droite  ({tot_l + tot_r} essais)")


if __name__ == "__main__":
    main()
