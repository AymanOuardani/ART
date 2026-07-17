"""
Reconstruit les essais NETTOYES vers la forme BRUTE (64 canaux, 160 Hz), sauve
dans Code/Final/, puis les visualise pour comparer avec le brut.

    python visualize_finale.py --subject 1 --model ART

Le finale = essais bruts ou seuls les 30 canaux ART sont remplaces par le signal
nettoye (resample 256 -> 160 Hz). Les 34 autres canaux gardent le brut.
"""

import argparse
from pathlib import Path
import numpy as np
import mne
from scipy.signal import resample_poly
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

RAW_DIR = Path(__file__).resolve().parent / "Brut"
NETTOYE = Path(__file__).resolve().parent / "Nettoyé"
OUT_DIR = Path(__file__).resolve().parent / "Final"
RUNS = [4, 8, 12]
EPOCH_LEN = 640           # 4 s a 160 Hz
MODELS = ["ICUNet", "ICUNet++", "ICUNet_attn", "ART"]

ART_TEMPLATE = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz",
                "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8", "TP7", "CP3", "CPz",
                "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]


def load_raw(subject):
    # Telecharge (dans Code/Brut), concatene et standardise les runs d'un sujet
    fnames = eegbci.load_data(subjects=[subject], runs=RUNS, path=str(RAW_DIR),
                              update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames],
                           verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw


def build_final(subject, model):
    # Essais bruts (64 canaux) avec les 30 canaux ART remplaces par le nettoye
    clean = mne.read_epochs(NETTOYE / f"S{subject:03d}" / f"{model}-epo.fif", verbose="ERROR")
    clean160 = resample_poly(clean.get_data(copy=True), 160, 256, axis=2)  # 256 -> 160 Hz

    raw = load_raw(subject)
    data = raw.get_data()
    art_idx = [raw.ch_names.index(n) for n in ART_TEMPLATE]
    events, eid = mne.events_from_annotations(raw, verbose="ERROR")
    labels = {eid["T1"]: 0, eid["T2"]: 1}

    X, y, k = [], [], 0
    for onset, _, code in events:
        if code not in labels:
            continue
        seg = data[:, onset:onset + EPOCH_LEN]
        if seg.shape[1] != EPOCH_LEN or k >= len(clean160):
            continue
        seg = seg.copy()
        seg[art_idx, :] = clean160[k]              # 30 canaux ART <- nettoye
        X.append(seg)
        y.append(labels[code])
        k += 1

    info = mne.create_info(raw.ch_names, 160.0, "eeg")
    info.set_montage("standard_1005", on_missing="ignore")
    events_arr = np.column_stack([np.arange(len(y)), np.zeros(len(y), int), np.array(y) + 1])
    return mne.EpochsArray(np.array(X), info, events_arr, tmin=0,
                           event_id={"gauche": 1, "droite": 2}, verbose="ERROR")


def main():
    ap = argparse.ArgumentParser(description="Reconstruit et visualise le finale d'un sujet.")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--model", choices=MODELS, default="ART")
    ap.add_argument("--label", choices=["gauche", "droite"], help="filtre une classe")
    args = ap.parse_args()

    epochs = build_final(args.subject, args.model)

    subj_dir = OUT_DIR / f"S{args.subject:03d}"
    subj_dir.mkdir(parents=True, exist_ok=True)
    epochs.save(subj_dir / f"{args.model}-epo.fif", overwrite=True)

    if args.label:
        epochs = epochs[args.label]

    print(f"Sujet {args.subject} ({args.model}) finale : {len(epochs)} essais, "
          f"{len(epochs.ch_names)} canaux, {epochs.info['sfreq']:.0f} Hz")

    color = {1: "#7fb3e0", 2: "#f08a87"}
    n_ch = len(epochs.ch_names)
    epoch_colors = [[color[code]] * n_ch for code in epochs.events[:, 2]]

    mne.viz.set_browser_backend("matplotlib")
    epochs.plot(title=f"Sujet {args.subject} - FINALE {args.model} - gauche (bleu) / droite (rouge)",
                epoch_colors=epoch_colors, block=True)


if __name__ == "__main__":
    main()
