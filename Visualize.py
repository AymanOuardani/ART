"""
Visualisation des essais EEG d'un sujet, a n'importe quelle etape du pipeline.
Fichier unique fusionnant les 4 anciens scripts visualize_* (etape positionnelle).

    python Visualize.py <etape> [sujet] [modele] [base] [--label gauche|droite]

    etape  : brut | pretraite | nettoye
    sujet  : numero du sujet            (defaut : 1)
    modele : ICUNet | ICUNet++ | ICUNet_attn | ART | DuoCL | GCTNet  (nettoye ; defaut ART)
    base   : original | EEGdenoiseNet   (nettoye ; defaut EEGdenoiseNet)

Exemples :
    python Visualize.py brut 1
    python Visualize.py pretraite 1
    python Visualize.py nettoye 1 DuoCL EEGdenoiseNet
    python Visualize.py nettoye 1 ART original --label gauche

Sources : brut = Databases/EEGBCI/Brut (64 ch, 160 Hz) ; pretraite = Output/Prétraité ;
nettoye = Output/Nettoyé/S###/{modele}_{base}-epo.fif (produit par Clean.py).
"""

import argparse
from pathlib import Path

import numpy as np
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

# ----------------------------- Chemins -----------------------------------
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "Databases" / "EEGBCI" / "Brut"
PRETRAITE = BASE / "Output" / "Prétraité"
NETTOYE = BASE / "Output" / "Nettoyé"

RUNS = [4, 8, 12]
EPOCH_LEN = 640            # 4 s a 160 Hz
MODELS = ["ICUNet", "ICUNet++", "ICUNet_attn", "ART", "DuoCL", "GCTNet"]
DATASETS = ["original", "EEGdenoiseNet"]


# ----------------------------- Brut ----------------------------------------
def load_raw(subject):
    # Telecharge (dans Databases/EEGBCI/Brut), concatene et standardise les runs d'un sujet
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


# ----------------------------- Affichage -----------------------------------
def show(epochs, subject, title, label):
    # Filtre eventuel par classe, affiche les infos, puis trace (gauche=bleu / droite=rouge).
    if label:
        epochs = epochs[label]
    print(f"Sujet {subject} : {len(epochs)} essais, {len(epochs.ch_names)} canaux, "
          f"{epochs.info['sfreq']:.0f} Hz")
    color = {1: "#7fb3e0", 2: "#f08a87"}
    n_ch = len(epochs.ch_names)
    epoch_colors = [[color[code]] * n_ch for code in epochs.events[:, 2]]
    mne.viz.set_browser_backend("matplotlib")
    epochs.plot(title=title, epoch_colors=epoch_colors, block=True)


# --------------------------------- Main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualise les essais d'un sujet a une etape du pipeline.")
    ap.add_argument("stage", choices=["brut", "pretraite", "nettoye"],
                    help="etape a visualiser")
    ap.add_argument("subject", nargs="?", type=int, default=1, help="numero du sujet (defaut : 1)")
    ap.add_argument("model", nargs="?", choices=MODELS, default="ART",
                    help="modele (nettoye ; defaut : ART)")
    ap.add_argument("dataset", nargs="?", choices=DATASETS, default="EEGdenoiseNet",
                    help="base d'entrainement du modele (nettoye ; defaut : EEGdenoiseNet)")
    ap.add_argument("--label", choices=["gauche", "droite"], help="filtre une classe")
    args = ap.parse_args()

    s = args.subject
    if args.stage == "brut":
        epochs = brut_epochs(s)
        title = f"Sujet {s} - BRUT - gauche (bleu) / droite (rouge)"

    elif args.stage == "pretraite":
        epochs = mne.read_epochs(PRETRAITE / f"S{s:03d}-epo.fif", verbose="ERROR")
        title = f"Sujet {s} - PRÉTRAITÉ - gauche (bleu) / droite (rouge)"

    else:  # nettoye
        d = NETTOYE / f"S{s:03d}"
        path = d / f"{args.model}_{args.dataset}-epo.fif"      # nommage Clean.py
        if not path.exists():
            old = d / f"{args.model}-epo.fif"                  # repli : ancien nommage
            if old.exists():
                path = old
            else:
                raise SystemExit(
                    f"ERREUR : essai debruite introuvable ({path.name}).\n"
                    f"  Lance d'abord : python Clean.py {args.model} {args.dataset}")
        epochs = mne.read_epochs(path, verbose="ERROR")
        title = f"Sujet {s} - {args.model} ({args.dataset}) - gauche (bleu) / droite (rouge)"

    show(epochs, s, title, args.label)


if __name__ == "__main__":
    main()
