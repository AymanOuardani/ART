"""
Fig 6A : carte topographique de la puissance bande mu (8-12 Hz) sur 30 canaux,
pendant l'imagerie motrice MAIN DROITE, pour les 3 conditions cote a cote :
  sans traitement (Prétraité) | ICUNet | ART (Nettoyé/S###/{model}).

Puissance en delta dB (echelle +/-3, comme l'article), spectre calcule en MNE
(epochs.compute_psd welch, n_fft=256). Une seule figure comparative.

    python topomap_mu.py    ->    Code/topomap_mu_comparaison.png
"""

from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")

BASE = Path(__file__).resolve().parent
PRETRAITE = BASE / "Prétraité"
NETTOYE = BASE / "Nettoyé"
SUBJECTS = range(1, 110)
NFFT = 256
MU = (8.0, 12.0)
VLIM = (-3.0, 3.0)
CONDITIONS = [("sans traitement", None), ("ICUNet", "ICUNet"), ("ART", "ART")]


def condition_db(model):
    # puissance mu en delta dB par canal (30,) + info, poolee sur les 109 sujets
    all_ep = []
    for s in SUBJECTS:
        f = (PRETRAITE / f"S{s:03d}-epo.fif" if model is None
             else NETTOYE / f"S{s:03d}" / f"{model}-epo.fif")
        if f.exists():
            all_ep.append(mne.read_epochs(f, verbose="ERROR")["droite"])
    epochs = mne.concatenate_epochs(all_ep, verbose="ERROR")
    spectrum = epochs.compute_psd(method="welch", n_fft=NFFT,
                                  fmin=MU[0], fmax=MU[1], verbose="ERROR")
    power = spectrum.get_data().mean(axis=(0, 2))
    return 10 * np.log10(power / power.mean()), spectrum.info


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    im = None
    for ax, (name, model) in zip(axes, CONDITIONS):
        db, info = condition_db(model)
        im, _ = mne.viz.plot_topomap(db, info, axes=ax, cmap="RdBu_r", vlim=VLIM,
                                     contours=6, show=False, sensors=True)
        ax.set_title(name, fontsize=13)
    fig.suptitle("Puissance mu (8-12 Hz) — imagerie main droite  (delta dB +/-3)",
                 fontsize=13)
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, location="right")
    cbar.set_label("delta dB (vs moyenne)")
    out = BASE / "topomap_mu_comparaison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
