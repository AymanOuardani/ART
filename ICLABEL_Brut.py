"""
Produit le signal de référence du projet : une ICA (infomax étendu) décomposée sur le signal
CONTINU à 64 canaux, dont ICLabel trie les composantes. Tout ce qui n'est classé ni "brain"
ni "other" est retiré, puis le signal reconstruit est enregistré.

  python ICLABEL_Brut.py      tous les sujets, reprend là où il s'est arrêté

Sortie : Databases/EEGBCI_ICLABEL/SXXX-raw.fif, à prétraiter ensuite avec
Pretraitement.py --iclabel pour obtenir Output/Nettoyé/SXXX/ICLABEL.fif.
"""

import pathlib as pl
import mne as mne
from mne.datasets import eegbci
from mne.preprocessing import ICA
from mne_icalabel import label_components

mne.set_log_level("ERROR")

#Chemins des fichiers : on nettoie les signaux continus, avant tout prétraitement
Brut_fif = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI")
Sortie = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI_ICLABEL")
Sortie.mkdir(parents=True, exist_ok=True)

#Traitement automatique de TOUS les sujets
for s in range(1, 110):
    sujet_id = "S" + str(s).zfill(3)
    fif_initial = Brut_fif / (sujet_id + "-raw.fif")
    out = Sortie / (sujet_id + "-raw.fif")
    if out.exists():        # reprise : on ne recalcule pas ce qui est déjà fait
        continue
    if not fif_initial.exists():
        continue
    raw = mne.io.read_raw_fif(fif_initial, preload=True)
    eegbci.standardize(raw)                          # Fc5. -> FC5
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.set_eeg_reference("average")                 # référence attendue par ICLabel

    #ICA infomax étendu (méthode requise par ICLabel), décomposition complète.
    #n_components = rang effectif et non nombre de canaux : la référence moyenne annule
    #une dimension (64 -> 63), et demander une composante de plus rend le blanchiment
    #dégénéré (variance ~1e-31) et détruit le signal à la reconstruction.
    rang = mne.compute_rank(raw, rank=None)["eeg"]
    ica = ICA(n_components=rang, method="infomax", fit_params=dict(extended=True),
              random_state=42, max_iter="auto")
    #passe-haut 1 Hz seul : le raw est à 160 Hz, donc Nyquist à 80 Hz et la borne haute
    #de 100 Hz d'ICLabel est inatteignable
    raw_filtre = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(raw_filtre)

    #Sélection ICLabel : on retire tout ce qui n'est pas "brain" ni "other"
    labels = label_components(raw_filtre, ica, method="iclabel")
    ica.exclude = [i for i, c in enumerate(labels["labels"]) if c not in ("brain", "other")]

    #Application de l'ICA et sauvegarde du signal continu nettoyé
    ica.apply(raw)
    raw.save(out, overwrite=True)
    print(sujet_id, ": composantes retirées", ica.exclude, "(", len(ica.exclude), "/", rang, ")", flush=True)

print("Termine : tous les sujets nettoyés (ICLabel sur le signal brut) dans", Sortie)
