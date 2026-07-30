import argparse as ap
import pathlib as pl
import mne as mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

mne.set_log_level("ERROR")

#Chemins des fichiers
Output = pl.Path(r"C:\Users\aymen\Desktop\ART\Output")

#Les deux jeux de runs d'imagerie motrice (cf. Pretraitement.py), chacun avec ses dossiers
JEUX = {"4812": "", "61014": "_FH"}

#Ligne de commande : jeu de runs
parser = ap.ArgumentParser(description="Nettoyage ICLabel de tous les sujets")
parser.add_argument("Runs", choices=list(JEUX),
                    help="4812 : main gauche / main droite | 61014 : les deux poings / les deux pieds")
args = parser.parse_args()

suffixe = JEUX[args.Runs]
Pretraite_Total = Output / ("Prétraité" + suffixe + "_Total")
Nettoye = Output / ("Nettoyé" + suffixe)

#Traitement automatique de TOUS les sujets
for s in range(1, 110):
    sujet_id = "S" + str(s).zfill(3)
    fif_initial = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
    if not fif_initial.exists():
        continue
    epochs = mne.read_epochs(fif_initial, preload=True)

    #ICA infomax étendu (méthode requise par ICLabel)
    ica = ICA(n_components=30, method="infomax", fit_params=dict(extended=True),
              random_state=42, max_iter="auto")
    epochs_filtres = epochs.copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(epochs_filtres)

    #Sélection ICLabel : on retire tout ce qui n'est pas "brain" ni "other"
    labels = label_components(epochs_filtres, ica, method="iclabel")
    ica.exclude = [i for i, c in enumerate(labels["labels"]) if c not in ("brain", "other")]

    #Application de l'ICA et sauvegarde des epochs nettoyés
    out = Nettoye / sujet_id / "ICLABEL.fif"
    out.parent.mkdir(parents=True, exist_ok=True)
    ica.apply(epochs)
    epochs.save(out, overwrite=True)
    print(sujet_id, ": composantes retirées", ica.exclude, "(", len(ica.exclude), "/30 )", flush=True)

print("Termine : tous les sujets nettoyés (ICLabel) dans", Nettoye)
