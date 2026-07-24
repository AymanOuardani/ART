import pathlib as pl
import mne as mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

mne.set_log_level("ERROR")

#Chemins des fichiers
Brut = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI\Brut")
Sortie = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI_fif")
Sortie.mkdir(parents=True, exist_ok=True)

runs = [4, 8, 12]          # imagerie motrice : main gauche (T1) / droite (T2)

#Pour chaque sujet : télécharge/charge les runs, les concatène, sauve en un seul .fif
for s in range(1, 110):
    fnames = eegbci.load_data(subjects=[s], runs=runs, path=str(Brut), update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in fnames])
    out = Sortie / ("S" + str(s).zfill(3) + "-raw.fif")
    raw.save(out, overwrite=True)
    print("Sujet", s, "regroupé ->", out.name)

print("Termine : 109 sujets regroupés en .fif dans", Sortie)
