"""
Produit le signal pseudo-propre : une ICA décomposée sur le signal CONTINU à 64 canaux,
dont ICLabel trie les composantes. Seules celles classées "brain" avec une confiance
supérieure à 80 % sont conservées, comme dans l'entraînement d'origine d'ART ; tout le reste
est retiré. Le signal reconstruit est ensuite enregistré.

  python ICLABEL_Brut.py              tous les sujets
  python ICLABEL_Brut.py 1-20         sujets 1 à 20
  python ICLABEL_Brut.py 1-3,5-109    tous sauf le sujet 4

Sortie : Databases/EEGBCI_ICLABEL/SXXX-raw.fif, à prétraiter ensuite avec
Pretraitement.py iclabel <sujet>.

Un sujet déjà nettoyé est ignoré : pour le refaire, supprimer sa sortie. L'ICA ajustée est
gardée dans Fitted_ICA/, si bien que changer le critère de tri ne demande pas de la
recalculer — c'est elle qui prend le temps.
"""

"""
English summary: builds the pseudo-clean reference signal used to train/evaluate ART:
fits (or reloads) an ICA on the continuous 64-channel raw signal, keeps only the
components ICLabel classifies as "brain" with confidence > 80%, reconstructs the signal
from those components, and saves it.

Usage:
  python ICLABEL_Brut.py [subjects]
  <subjects> is a range/list like "1-109" (default), "1,2,5" or "1-3,5-109".
Output: Databases/EEGBCI_ICLABEL/SXXX-raw.fif (subjects already processed are skipped).
"""

import argparse as ap
import pathlib as pl
import time
import mne as mne
from mne.datasets import eegbci
from mne.preprocessing import ICA, read_ica
from mne_icalabel import label_components

mne.set_log_level("ERROR")

#Chemins des fichiers
Racine = pl.Path(__file__).resolve().parent
Brut_fif = Racine / "Databases" / "EEGBCI"
Sortie = Racine / "Databases" / "EEGBCI_ICLABEL"
Fitted_ICA = Racine / "Fitted_ICA"

#Une composante n'est gardée que si ICLabel la dit cérébrale avec au moins cette confiance
Seuil_brain = 0.80

#Ligne de commande
parser = ap.ArgumentParser(description="Nettoyage ICLabel du signal continu (ICA sur 64 canaux)")
parser.add_argument("Sujets", nargs="?", default="1-109",
                    help="ex. 1-109, 1,2,5 ou 1-3,5-109 (défaut : tous)")
args = parser.parse_args()

#Liste de sujets
sujets = []
for bloc in args.Sujets.split(","):
    if "-" in bloc:
        a, b = bloc.split("-")
        sujets += list(range(int(a), int(b) + 1))
    else:
        sujets.append(int(bloc))

Sortie.mkdir(parents=True, exist_ok=True)
Fitted_ICA.mkdir(parents=True, exist_ok=True)

for s in sujets:
    sujet_id = "S" + str(s).zfill(3)
    entree = Brut_fif / (sujet_id + "-raw.fif")
    out = Sortie / (sujet_id + "-raw.fif")
    if out.exists():
        print(sujet_id, ": déjà nettoyé", flush=True)
        continue
    if not entree.exists():
        print(sujet_id, ": introuvable", entree, flush=True)
        continue

    debut = time.time()
    raw = mne.io.read_raw_fif(entree, preload=True)
    eegbci.standardize(raw)                          # Fc5. -> FC5
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.set_eeg_reference("average")                 # référence attendue par ICLabel

    #Passe-haut 1 Hz seul : à 160 Hz, la borne haute d'ICLabel est inatteignable
    raw_filtre = raw.copy().filter(l_freq=1.0, h_freq=None)

    #ICA infomax étendu, exigée par ICLabel. n_components = rang effectif et non nombre de
    #canaux : la référence moyenne annule une dimension, et en demander une de plus rend le
    #blanchiment dégénéré. L'ICA est gardée sur disque, c'est elle qui prend le temps.
    ica_file = Fitted_ICA / (sujet_id + "-raw-ica.fif")
    if ica_file.exists():
        ica = read_ica(ica_file)
    else:
        rang = mne.compute_rank(raw, rank=None)["eeg"]
        ica = ICA(n_components=rang, method="infomax", fit_params=dict(extended=True),
                  random_state=42, max_iter="auto")
        ica.fit(raw_filtre)
        ica.save(ica_file)

    #On ne garde que les composantes cérébrales sûres
    labels = label_components(raw_filtre, ica, method="iclabel")
    ica.exclude = [i for i, (c, p) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
                   if not (c == "brain" and p > Seuil_brain)]

    #Application et sauvegarde
    ica.apply(raw)
    raw.save(out, overwrite=True)
    gardees = ica.n_components_ - len(ica.exclude)
    print(f"{sujet_id} : {gardees}/{ica.n_components_} composantes gardées "
          f"({len(ica.exclude)} retirées) - {time.time() - debut:.0f}s", flush=True)

print("Terminé :", len(sujets), "sujets demandés ->", Sortie)
