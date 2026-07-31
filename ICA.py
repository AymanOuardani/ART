import argparse as ap
import pathlib as pl
import mne as mne
from mne.preprocessing import ICA, read_ica

#Chemins des fichiers
Pretraite_Total = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité_Total")
Fitted_ICA = pl.Path(r"C:\Users\aymen\Desktop\ART\Fitted_ICA")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")

#Ligne de commande pour récupérer le numéro du sujet
parser = ap.ArgumentParser(description="ICA")
parser.add_argument("Sujet", type=int, help='Numéro du sujet (1-109)')
args = parser.parse_args()

#Lecture des epochs
sujet_id = "S" + str(args.Sujet).zfill(3)
fif_initial = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
epochs = mne.read_epochs(fif_initial, preload=True)

#Lecture/Fitting de l'ICA
ica_file = Fitted_ICA / (sujet_id + "-ica.fif")
if ica_file.exists():
    ica = read_ica(ica_file)
    print(f"ICA existe déjà pour Sujet {args.Sujet}. Lecture réussie.")
else:
    print(f"ICA n'existe pas encore pour Sujet {args.Sujet}. Fitting en cours...")
    ica = ICA(n_components=30, method="fastica", fit_params=dict(extended=True),
              random_state=42, max_iter="auto")
    epochs_filtres = epochs.copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(epochs_filtres)
    ica.save(ica_file)

#Plot des sources ICA
ica.plot_sources(epochs, block=True)
print("Composantes retirées : ", ica.exclude)

#Application de l'ICA et sauvegarde des epochs nettoyés
out = Nettoye / sujet_id / "ICA.fif"
out.parent.mkdir(parents=True, exist_ok=True)
ica.apply(epochs)
epochs.save(out, overwrite=True)
print("Signaux nettoyés sauvegardés pour Sujet ", args.Sujet)