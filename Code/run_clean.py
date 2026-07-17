"""
Nettoie tous les sujets avec tous les modeles, en attendant les fichiers
prétraités s'ils ne sont pas encore la (l'autre terminal les produit).

    python run_clean.py --subjects 1-109
"""

import argparse
import time
from pathlib import Path
import numpy as np
import mne

import utils

mne.set_log_level("ERROR")
BASE = Path(__file__).resolve().parent
PRETRAITE = BASE / "Prétraité"
NETTOYE = BASE / "Nettoyé"
MODELS = ["ICUNet", "ICUNet++", "ICUNet_attn", "ART"]


def wait_epochs(path, wait=30):
    # Attend que le fichier existe ET soit lisible (ecriture terminee)
    while True:
        if path.exists():
            try:
                return mne.read_epochs(path, verbose="ERROR")
            except Exception:
                pass  # ecriture en cours -> on reessaie
        print(f"  ... en attente de {path.name}", flush=True)
        time.sleep(wait)


def clean_subject(subject):
    epochs = wait_epochs(PRETRAITE / f"S{subject:03d}-epo.fif")
    data = epochs.get_data(copy=True)
    subj_dir = NETTOYE / f"S{subject:03d}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        out = subj_dir / f"{model}-epo.fif"
        if out.exists():
            continue
        cleaned = np.stack([utils.clean_epoch(ep, model) for ep in data])
        mne.EpochsArray(cleaned, epochs.info, epochs.events, tmin=epochs.tmin,
                        event_id=epochs.event_id, metadata=epochs.metadata,
                        verbose="ERROR").save(out, overwrite=True)
        print(f"  S{subject:03d} {model:12s} -> ok", flush=True)


def parse_subjects(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description="Nettoyage batch de tous les sujets.")
    ap.add_argument("--subjects", default="1-109")
    args = ap.parse_args()

    for s in parse_subjects(args.subjects):
        print(f"Sujet {s}", flush=True)
        clean_subject(s)
    print("Termine.", flush=True)


if __name__ == "__main__":
    main()
