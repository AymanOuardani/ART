"""
Produit le signal pseudo-propre : une ICA décomposée sur le signal CONTINU à 64 canaux,
dont ICLabel trie les composantes. Seules celles classées "brain" avec une confiance
supérieure à 80 % sont conservées, comme dans l'entraînement d'origine d'ART ; tout le reste
est retiré. Le signal reconstruit est ensuite enregistré.

  python ICLABEL_Brut.py                tous les sujets
  python ICLABEL_Brut.py 1-20           sujets 1 à 20
  python ICLABEL_Brut.py --sauf 4       tous sauf le sujet 4
  python ICLABEL_Brut.py --force        refait même si la sortie existe déjà

Sortie : Databases/EEGBCI_ICLABEL/SXXX-raw.fif, à prétraiter ensuite avec
Pretraitement.py <sujet> --iclabel.

L'ICA ajustée est gardée dans Fitted_ICA/ : changer de critère de tri ne demande plus de la
recalculer, c'est la partie longue.
"""

import argparse as ap
import pathlib as pl
import time
import mne as mne
from mne.datasets import eegbci
from mne.preprocessing import ICA, read_ica
from mne_icalabel import label_components

mne.set_log_level("ERROR")

#Chemins des fichiers : on nettoie les signaux continus, avant tout prétraitement
Brut_fif = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI")
Sortie = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI_ICLABEL")
Fitted_ICA = pl.Path(r"C:\Users\aymen\Desktop\ART\Fitted_ICA")

#Critère de sélection : une composante n'est gardée que si ICLabel la dit cérébrale avec au
#moins cette confiance. Tout le reste (oeil, muscle, ligne, coeur, "other") est retiré.
SEUIL_BRAIN = 0.80

#Ligne de commande : quels sujets traiter
parser = ap.ArgumentParser(description="Nettoyage ICLabel du signal continu (ICA sur 64 canaux)")
parser.add_argument("Sujets", nargs="?", default="1-109", help="ex. 1-109 ou 1,2,5 (défaut : tous)")
parser.add_argument("--sauf", default=None, help="sujets à ne pas traiter, ex. 4 ou 4,7")
parser.add_argument("--force", action="store_true", help="refait même si la sortie existe déjà")
args = parser.parse_args()

#"1-109" ou "1,2,5" -> liste de sujets, moins ceux de --sauf
if "-" in args.Sujets:
    a, b = args.Sujets.split("-")
    sujets = list(range(int(a), int(b) + 1))
else:
    sujets = [int(s) for s in args.Sujets.split(",")]
if args.sauf:
    exclus = [int(s) for s in args.sauf.split(",")]
    sujets = [s for s in sujets if s not in exclus]
    print("Sujets écartés :", exclus)

Sortie.mkdir(parents=True, exist_ok=True)
Fitted_ICA.mkdir(parents=True, exist_ok=True)

for s in sujets:
    sujet_id = "S" + str(s).zfill(3)
    entree = Brut_fif / (sujet_id + "-raw.fif")
    out = Sortie / (sujet_id + "-raw.fif")
    if out.exists() and not args.force:
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

    #passe-haut 1 Hz seul : le raw est à 160 Hz, donc Nyquist à 80 Hz et la borne haute
    #de 100 Hz d'ICLabel est inatteignable
    raw_filtre = raw.copy().filter(l_freq=1.0, h_freq=None)

    #ICA infomax étendu (méthode requise par ICLabel), décomposition complète.
    #n_components = rang effectif et non nombre de canaux : la référence moyenne annule
    #une dimension (64 -> 63), et demander une composante de plus rend le blanchiment
    #dégénéré (variance ~1e-31) et détruit le signal à la reconstruction.
    #L'ICA est gardée sur disque : c'est elle qui coûte du temps, pas le tri.
    ica_file = Fitted_ICA / (sujet_id + "-raw-ica.fif")
    if ica_file.exists():
        ica = read_ica(ica_file)
    else:
        rang = mne.compute_rank(raw, rank=None)["eeg"]
        ica = ICA(n_components=rang, method="infomax", fit_params=dict(extended=True),
                  random_state=42, max_iter="auto")
        ica.fit(raw_filtre)
        ica.save(ica_file)

    #Sélection : on ne garde que les composantes cérébrales sûres, on retire tout le reste
    labels = label_components(raw_filtre, ica, method="iclabel")
    ica.exclude = [i for i, (c, p) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
                   if not (c == "brain" and p > SEUIL_BRAIN)]

    #Application de l'ICA et sauvegarde du signal continu nettoyé
    ica.apply(raw)
    raw.save(out, overwrite=True)
    gardees = ica.n_components_ - len(ica.exclude)
    print(f"{sujet_id} : {gardees}/{ica.n_components_} composantes gardées "
          f"({len(ica.exclude)} retirées) - {time.time() - debut:.0f}s", flush=True)

print("Terminé :", len(sujets), "sujets demandés ->", Sortie)
