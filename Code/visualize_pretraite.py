"""
Visualise les essais pretraites (fichier .fif) d'un sujet.

    python visualize_pretraite.py --subject 1
    python visualize_pretraite.py --subject 1 --label gauche   # que la main gauche

Lit Code/Prétraité/S{n}-epo.fif (produit par Pretreatement.py).
"""

import argparse
from pathlib import Path
import mne

OUT_DIR = Path(__file__).resolve().parent / "Prétraité"


def main():
    ap = argparse.ArgumentParser(description="Visualise les essais pretraites d'un sujet.")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--label", choices=["gauche", "droite"], help="filtre une classe")
    args = ap.parse_args()

    path = OUT_DIR / f"S{args.subject:03d}-epo.fif"
    epochs = mne.read_epochs(path, verbose="ERROR")
    if args.label:
        epochs = epochs[args.label]

    print(f"Sujet {args.subject} : {len(epochs)} essais, {len(epochs.ch_names)} canaux, "
          f"{epochs.info['sfreq']:.0f} Hz")

    # Couleur douce par essai : gauche = bleu, droite = rouge
    color = {1: "#7fb3e0", 2: "#f08a87"}
    n_ch = len(epochs.ch_names)
    epoch_colors = [[color[code]] * n_ch for code in epochs.events[:, 2]]

    mne.viz.set_browser_backend("matplotlib")
    epochs.plot(title=f"Sujet {args.subject} - gauche (bleu) / droite (rouge)",
                epoch_colors=epoch_colors, block=True)


if __name__ == "__main__":
    main()
