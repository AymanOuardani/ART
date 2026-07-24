import pathlib as pl
import mne as mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

mne.set_log_level("ERROR")

#Chemins des fichiers
Pretraite_Total = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité_Total")
Nettoye_ICLABEL = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé_ICLABEL")

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
    out = Nettoye_ICLABEL / sujet_id / (sujet_id + "-ICA.fif")
    out.parent.mkdir(parents=True, exist_ok=True)
    ica.apply(epochs)
    epochs.save(out, overwrite=True)
    print(sujet_id, ": composantes retirées", ica.exclude, "(", len(ica.exclude), "/30 )", flush=True)

print("Termine : tous les sujets nettoyés (ICLabel) dans", Nettoye_ICLABEL)
