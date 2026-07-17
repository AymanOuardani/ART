"""
Explore un sujet de la dataset EEGBCI (BCI2000) : infos + visualisation.

    python Dataset.py --subject 1                    # infos
    python Dataset.py --subject 1 --plot             # infos + signaux
    python Dataset.py --subject 5 --runs 6 10 14     # autres runs

Runs 1/2    = baseline yeux ouverts / fermes.
Runs 3/7/11 = mouvement reel main gauche (T1) / droite (T2).
Runs 4/8/12 = motor imagery main gauche (T1) / droite (T2).
Runs 5/9/13 = mouvement reel deux poings (T1) / deux pieds (T2).
Runs 6/10/14 = motor imagery deux poings (T1) / deux pieds (T2).
"""

import argparse
from pathlib import Path
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

RAW_DIR = Path(__file__).resolve().parent / "Brut"   # EDF bruts telecharges ici


def load_subject(subject, runs):
    # Telecharge (dans Code/Brut), concatene et standardise les runs d'un sujet
    fnames = eegbci.load_data(subjects=[subject], runs=runs, path=str(RAW_DIR),
                              update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames],
                           verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw


def annot_labels(runs):
    # Sens de T1/T2 selon le type de runs choisi
    if set(runs) <= {3, 4, 7, 8, 11, 12}:
        t1, t2 = "main gauche", "main droite"
    elif set(runs) <= {5, 6, 9, 10, 13, 14}:
        t1, t2 = "deux poings", "deux pieds"
    else:
        t1, t2 = "T1", "T2"
    return {"T0": "repos", "T1": t1, "T2": t2}


def print_info(raw, subject, runs):
    # Affiche un resume lisible du sujet
    print("=" * 60)
    print(f"Sujet {subject}  |  runs {runs}")
    print("=" * 60)
    print(f"  Canaux         : {raw.info['nchan']}")
    print(f"  Frequence      : {raw.info['sfreq']:.1f} Hz")
    print(f"  Echantillons   : {raw.n_times}  ({raw.n_times / raw.info['sfreq']:.1f} s)")
    print(f"  Canaux (debut) : {', '.join(raw.ch_names[:8])} ...")

    print("\n  Evenements :")
    labels = annot_labels(runs)
    counts = {}
    for d in raw.annotations.description:
        if "boundary" not in d:
            counts[d] = counts.get(d, 0) + 1
    for code in sorted(counts):
        print(f"    {code} x{counts[code]:<4d} -> {labels.get(code, '?')}")

    data_uv = raw.get_data() * 1e6
    print("\n  Signal (uV) :")
    print(f"    min / max      : {data_uv.min():.1f} / {data_uv.max():.1f} (minimum / maximum sur tous les canaux)")
    print(f"    ecart-type moy : {data_uv.std(axis=1).mean():.1f} (écart-type moyen sur tous les canaux)")
    print("=" * 60)


def plot_signals(raw, subject, runs, duration):
    # Ouvre la fenetre MNE avec les signaux et les evenements
    mne.viz.set_browser_backend("matplotlib")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    raw.plot(title=f"Sujet {subject} - runs {runs}", duration=duration,
             n_channels=min(30, raw.info["nchan"]),
             events=events, event_id=event_id, block=True)


def main():
    ap = argparse.ArgumentParser(description="Explore un sujet EEGBCI (BCI2000).")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    raw = load_subject(args.subject, args.runs)
    print_info(raw, args.subject, args.runs)
    if args.plot:
        plot_signals(raw, args.subject, args.runs, args.duration)


if __name__ == "__main__":
    main()
