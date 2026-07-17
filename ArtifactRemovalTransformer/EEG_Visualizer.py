"""
EEG_Visualizer.py
-----------------
Visualise l'EEG AVANT (sampledata) et APRES (outputsample) le debruitage ART,
dans deux fenetres MNE, a la MEME echelle et en MICROVOLTS.

sampledata.csv est deja en microvolts. outputsample.csv est la sortie brute du
modele : le modele z-score son entree en interne mais n'inverse jamais cette
normalisation, donc sa sortie est en z-score. On inverse donc le z-score sur la
sortie a l'aide de la moyenne et de l'ecart-type par canal calcules sur l'entree
microvolts :

    apres_uV = apres_zscore * std(avant) + mean(avant)

Les deux signaux sont ensuite affiches en volts (MNE attend des unites SI), a une
echelle commune.

Usage :
    python EEG_Visualizer.py
    python EEG_Visualizer.py --before sampledata/sampledata.csv \
                             --after  sampledata/outputsample.csv \
                             --loc    sampledata/sample_chanlocs.loc \
                             --sfreq  256

Dependances :
    pip install mne matplotlib "numpy<2" "scipy==1.11.4"
"""

import argparse
import csv
import numpy as np
import mne


def read_channel_names(loc_file):
    """Lit les noms de canaux depuis un fichier .loc (4e colonne)."""
    names = []
    with open(loc_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                names.append(parts[3])
    return names


def read_csv_matrix(csv_file):
    """Lit un CSV 'canaux x temps' -> array float64 de forme (n_canaux, n_temps)."""
    with open(csv_file, newline="") as f:
        rows = [row for row in csv.reader(f)]
    return np.array(rows).astype(np.float64)


def channel_stats(data):
    """Moyenne et ecart-type par canal. std=0 -> 1 pour eviter les divisions nulles."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return mean, std


def invert_zscore(data, mean, std):
    """Inverse le z-score du modele : ramene la sortie en microvolts."""
    return data * std + mean


def build_raw(data_uv, ch_names, sfreq, loc_file=None):
    """Construit un objet MNE RawArray a partir d'une matrice (canaux x temps) en uV.

    MNE attend des unites SI (volts) : on convertit donc les microvolts en volts.
    """
    n_ch = data_uv.shape[0]

    # ajuste la liste de noms a la taille reelle des donnees
    if len(ch_names) != n_ch:
        print(f"[!] {len(ch_names)} noms pour {n_ch} canaux -> noms generiques.")
        ch_names = [f"ch{i}" for i in range(n_ch)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")  # uV -> V

    # essaie d'appliquer le montage (positions des electrodes) depuis le .loc
    if loc_file is not None:
        try:
            montage = mne.channels.read_custom_montage(loc_file)
            raw.set_montage(montage, match_case=False, on_missing="ignore",
                            verbose="ERROR")
        except Exception as e:
            print(f"[!] montage non applique ({e}) -- affichage sans positions.")
    return raw


def main():
    ap = argparse.ArgumentParser(description="Visualiseur EEG avant/apres ART.")
    ap.add_argument("--before", default="sampledata/sampledata.csv")
    ap.add_argument("--after", default="sampledata/outputsample.csv")
    ap.add_argument("--loc", default="sampledata/sample_chanlocs.loc")
    ap.add_argument("--sfreq", type=float, default=256.0)
    args = ap.parse_args()

    # backend matplotlib (pas besoin de mne-qt-browser)
    mne.viz.set_browser_backend("matplotlib")

    ch_names = read_channel_names(args.loc)

    print(f"Lecture   AVANT : {args.before}")
    before = read_csv_matrix(args.before)
    print(f"  -> {before.shape[0]} canaux x {before.shape[1]} echantillons")

    print(f"Lecture   APRES : {args.after}")
    after = read_csv_matrix(args.after)
    print(f"  -> {after.shape[0]} canaux x {after.shape[1]} echantillons")

    # AVANT est deja en microvolts. APRES est en z-score (sortie brute du modele) :
    # on inverse le z-score avec les stats par canal de l'entree pour revenir en uV.
    mean, std = channel_stats(before)
    after = invert_zscore(after, mean, std)

    # echelle commune (uV), calculee sur les signaux centres pour ignorer le DC.
    before_c = before - before.mean(axis=1, keepdims=True)
    after_c = after - after.mean(axis=1, keepdims=True)
    scale_uv = max(np.percentile(np.abs(before_c), 99),
                   np.percentile(np.abs(after_c), 99))
    common_scalings = dict(eeg=float(scale_uv) * 1e-6)  # uV -> V
    print(f"Echelle commune : eeg = {scale_uv:.2f} uV")

    raw_before = build_raw(before, ch_names, args.sfreq, args.loc)
    raw_after = build_raw(after, ch_names, args.sfreq, args.loc)

    raw_before.plot(title="AVANT  -  sampledata (microvolts, echelle commune)",
                    duration=10, n_channels=min(30, before.shape[0]),
                    scalings=common_scalings, show=True, block=False)

    raw_after.plot(title="APRES  -  outputsample (microvolts, echelle commune)",
                   duration=10, n_channels=min(30, after.shape[0]),
                   scalings=common_scalings, show=True, block=True)


if __name__ == "__main__":
    main()
