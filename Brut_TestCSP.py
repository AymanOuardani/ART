import pathlib as pl
import time
import numpy as np
import mne
from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

mne.set_log_level("ERROR")

Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")
data_dir = pl.Path(r"Résultats\data")

sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))
N_iter = 10
n_components_liste = [2, 4, 6, 8, 10, 12, 14]

def prep(X):
    X = X - X.mean(axis=1, keepdims=True)
    X = filter_data(X, sfreq, fmin, fmax, verbose=False)
    return X[:, :, crop]

cv = ShuffleSplit(N_iter, test_size=0.2, random_state=42)

lignes = []
sum_par_nc = {nc: [] for nc in n_components_liste}
t_debut = time.time()

for i in range(1, 110):
    sujet_id = "S" + str(i).zfill(3)
    f = Pretraite / (sujet_id + "_Pre.fif")
    if not f.exists():
        continue
    epochs = mne.read_epochs(f, preload=True)["gauche", "droite"]   # retire le repos T0
    X = epochs.get_data()
    y = epochs.events[:, 2]
    Xp = prep(X)

    row = []
    for nc in n_components_liste:
        clf = Pipeline([("CSP", CSP(n_components=nc, reg=None, log=True, norm_trace=False)),
                        ("LDA", LinearDiscriminantAnalysis())])
        scores = cross_val_score(clf, Xp, y, cv=cv)
        acc_m, acc_s = scores.mean(), scores.std()
        row.append((acc_m, acc_s))
        sum_par_nc[nc].append(acc_m)

    lignes.append((sujet_id, row))
    print(f"{sujet_id} : " + " | ".join(f"nc{nc}={a:.2f}" for nc, (a, s) in zip(n_components_liste, row))
          + f"  (total {time.time()-t_debut:.0f}s)", flush=True)

#Construction du tableau LaTeX (gras sur la meilleure accuracy de chaque ligne, et sur la meilleure moyenne)
lignes_tex = []
for sujet_id, row in lignes:
    best_acc = max(a for a, s in row)
    cellules = []
    for a, s in row:
        txt = f"{a:.2f} $\\pm$ {s:.2f}"
        cellules.append(f"\\textbf{{{txt}}}" if a == best_acc else txt)
    lignes_tex.append(f"{sujet_id} & " + " & ".join(cellules) + r" \\")

moyennes = [np.mean(sum_par_nc[nc]) for nc in n_components_liste]
ecarts = [np.std(sum_par_nc[nc]) for nc in n_components_liste]
best_moy = max(moyennes)
cellules_moy = []
for m, e in zip(moyennes, ecarts):
    txt = f"{m:.2f} $\\pm$ {e:.2f}"
    cellules_moy.append(f"\\textbf{{{txt}}}" if m == best_moy else txt)
lignes_tex.append(r"\midrule \textbf{Moyenne} & " + " & ".join(cellules_moy) + r" \\")

out = data_dir / "recap_csp_ncomp.tex"
out.write_text("\\def\\recapcspncomp{\n" + "\n".join(lignes_tex) + "}\n", encoding="utf-8", newline="\n")
print("TOUT_TERMINE_CSP_NCOMP, n_sujets =", len(lignes))
