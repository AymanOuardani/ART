"""
Visualise les essais BRUTS d'un sujet (64 canaux, 160 Hz), pour comparer au finale.

    python visualize_brut.py --subject 1
    python visualize_brut.py --subject 1 --label gauche

Decoupe le brut en essais T1 (gauche) / T2 (droite) de 4 s (640 pts a 160 Hz).
"""

import argparse
from pathlib import Path
import numpy as np
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

RAW_DIR = Path(__file__).resolve().parent / "Brut"
RUNS = [4, 8, 12]
EPOCH_LEN = 640            # 4 s a 160 Hz


def load_raw(subject):
    # Telecharge (dans Code/Brut), concatene et standardise les runs d'un sujet
    fnames = eegbci.load_data(subjects=[subject], runs=RUNS, path=str(RAW_DIR),
                              update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames],
                           verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw


def brut_epochs(subject):
    # Decoupe le brut en essais (64 canaux, 160 Hz) autour de T1/T2
    raw = load_raw(subject)
    data = raw.get_data()
    events, eid = mne.events_from_annotations(raw, verbose="ERROR")
    labels = {eid["T1"]: 0, eid["T2"]: 1}

    X, y = [], []
    for onset, _, code in events:
        if code not in labels:
            continue
        seg = data[:, onset:onset + EPOCH_LEN]
        if seg.shape[1] == EPOCH_LEN:
            X.append(seg)
            y.append(labels[code])

    info = mne.create_info(raw.ch_names, 160.0, "eeg")
    info.set_montage("standard_1005", on_missing="ignore")
    events_arr = np.column_stack([np.arange(len(y)), np.zeros(len(y), int), np.array(y) + 1])
    return mne.EpochsArray(np.array(X), info, events_arr, tmin=0,
                           event_id={"gauche": 1, "droite": 2}, verbose="ERROR")


def main():
    ap = argparse.ArgumentParser(description="Visualise les essais bruts d'un sujet.")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--label", choices=["gauche", "droite"], help="filtre une classe")
    args = ap.parse_args()

    epochs = brut_epochs(args.subject)
    if args.label:
        epochs = epochs[args.label]

    print(f"Sujet {args.subject} (brut) : {len(epochs)} essais, "
          f"{len(epochs.ch_names)} canaux, {epochs.info['sfreq']:.0f} Hz")

    color = {1: "#7fb3e0", 2: "#f08a87"}
    n_ch = len(epochs.ch_names)
    epoch_colors = [[color[code]] * n_ch for code in epochs.events[:, 2]]

    mne.viz.set_browser_backend("matplotlib")
    epochs.plot(title=f"Sujet {args.subject} - BRUT - gauche (bleu) / droite (rouge)",
                epoch_colors=epoch_colors, block=True)


if __name__ == "__main__":
    main()
