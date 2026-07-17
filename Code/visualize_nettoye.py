"""
Visualise les essais nettoyes (fichier .fif) d'un sujet, pour un modele donne.

    python visualize_nettoye.py --subject 1 --model ART
    python visualize_nettoye.py --subject 1 --model ICUNet --label gauche

Lit Code/Nettoyé/S{n}/{model}-epo.fif (produit par main.py).
"""

import argparse
from pathlib import Path
import mne

OUT_DIR = Path(__file__).resolve().parent / "Nettoyé"
MODELS = ["ICUNet", "ICUNet++", "ICUNet_attn", "ART"]


def main():
    ap = argparse.ArgumentParser(description="Visualise les essais nettoyes d'un sujet.")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--model", choices=MODELS, default="ART")
    ap.add_argument("--label", choices=["gauche", "droite"], help="filtre une classe")
    args = ap.parse_args()

    path = OUT_DIR / f"S{args.subject:03d}" / f"{args.model}-epo.fif"
    epochs = mne.read_epochs(path, verbose="ERROR")
    if args.label:
        epochs = epochs[args.label]

    print(f"Sujet {args.subject} ({args.model}) : {len(epochs)} essais, "
          f"{len(epochs.ch_names)} canaux, {epochs.info['sfreq']:.0f} Hz")

    # Couleur douce par essai : gauche = bleu, droite = rouge
    color = {1: "#7fb3e0", 2: "#f08a87"}
    n_ch = len(epochs.ch_names)
    epoch_colors = [[color[code]] * n_ch for code in epochs.events[:, 2]]

    mne.viz.set_browser_backend("matplotlib")
    epochs.plot(title=f"Sujet {args.subject} - {args.model} - gauche (bleu) / droite (rouge)",
                epoch_colors=epoch_colors, block=True)


if __name__ == "__main__":
    main()
