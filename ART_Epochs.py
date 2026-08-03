"""
Suit ce qu'apprend ART au fil de son entraînement : chacun des 60 checkpoints d'un sujet est
appliqué à ses essais, puis évalué sur les deux critères qui comptent — le RMSE contre
ICLabel (fidélité à la cible) et l'accuracy CSP+LDA (information motrice conservée).

  python ART_Epochs.py 1-16                 les 60 epochs des sujets 1 à 16
  python ART_Epochs.py 4 --meilleure-seule  seulement la meilleure epoch de validation
  python ART_Epochs.py 4 --rapport-seul     réaffiche le tableau sans recalculer
  python ART_Epochs.py 4 --modele ART_v2    autre dossier d'entraînement dans Model/

Les résultats sont enregistrés après chaque epoch dans Output/ART/SXXX/resultats_epochs.json :
une exécution interrompue reprend sans rien recalculer. La date du checkpoint utilisé est
gardée avec chaque résultat, si bien qu'un réentraînement invalide automatiquement les valeurs
devenues périmées. Le signal de la meilleure epoch est conservé en .fif.
"""

import argparse as ap
import json
import pathlib as pl
import time

import numpy as np
import torch
import mne as mne
from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

import Utils

mne.set_log_level("ERROR")

#Chemins des fichiers
Output = pl.Path(r"C:\Users\aymen\Desktop\ART\Output")
Pretraite = Output / "Prétraité"                   # entrée bruitée (3 classes, repos inclus)
Nettoye = Output / "Nettoyé"
Sortie_ART = Output / "ART"                        # signal débruité de la meilleure epoch
Model_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Model")
Modele_Defaut = "ART_ICLABEL"                      # dossier des checkpoints LOSO dans Model/

#Signal de référence : l'ICA ICLabel appliquée au signal continu à 64 canaux (ICLABEL_Brut.py),
#c'est aussi la cible d'entraînement d'ART (cf. Train_Model.py)
Reference = "ICLABEL.fif"
N_EPOCHS = 60

#Paramètres CSP + LDA : strictement ceux d'Evaluation.py (runs 4/8/12, main gauche vs main droite)
sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))   # fenêtre [1,2]s (+ retard FIR ~1.95s)
N_iter = 10
clf = Pipeline([("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                ("LDA", LinearDiscriminantAnalysis())])
cv = ShuffleSplit(N_iter, test_size=0.2, random_state=42)

#Ligne de commande : quels sujets traiter
parser = ap.ArgumentParser(description="ART (LOSO) epoch par epoch : RMSE contre ICLabel et accuracy CSP+LDA")
parser.add_argument("Sujets", nargs="?", default="1-109", help="ex. 1-16 ou 1,2,5 (défaut : 1-109)")
parser.add_argument("--force", action="store_true", help="recalcule les epochs déjà enregistrées")
parser.add_argument("--meilleure-seule", action="store_true",
                    help="ne traite que la meilleure epoch de validation, sans les 59 autres")
parser.add_argument("--modele", default=Modele_Defaut,
                    help=f"dossier des checkpoints LOSO dans Model/ (défaut : {Modele_Defaut})")
args = parser.parse_args()

Modelsave = Model_Dir / args.modele / "modelsave"

#"1-16" ou "1,2,5" -> liste de sujets (même convention que Clean_Model.py)
if "-" in args.Sujets:
    a, b = args.Sujets.split("-")
    sujets = list(range(int(a), int(b) + 1))
else:
    sujets = [int(s) for s in args.Sujets.split(",")]


def prep(X):
    X = X - X.mean(axis=1, keepdims=True)              # référence moyenne
    X = filter_data(X, sfreq, fmin, fmax)              # band-pass 7-30 Hz
    return X[:, :, crop]                                # fenêtre [1,2]s


def lit_checkpoint(sujet_id, ep):
    # Chaque checkpoint porte son epoch et son RMSE de validation. On relève aussi la date du
    # fichier : Train_Model.py n'a pas de reprise, le relancer réécrit les checkpoints avec des
    # poids différents sans rien signaler, et il ne faut pas réutiliser un résultat périmé.
    f = Modelsave / sujet_id / f"Epoch_N{ep}" / "checkpoint.pth.tar"
    if not f.exists():
        return None
    c = torch.load(f, map_location="cpu", weights_only=False)
    return {"rmse_val": float(c["rmse_val"]), "modele": round(f.stat().st_mtime, 3)}


def meilleure_epoch_validation(sujet_id):
    # Meilleure epoch = RMSE de validation minimal, lu dans les checkpoints eux-mêmes
    val = {}
    for ep in range(1, N_EPOCHS + 1):
        c = lit_checkpoint(sujet_id, ep)
        if c is not None:
            val[ep] = c["rmse_val"]
    if not val:
        return None, None
    meilleure = min(val, key=val.get)
    return meilleure, val[meilleure]


def phrase_epochs(numeros):
    # "à l'epoch 8" / "à l'epoch 49 et à l'epoch 58" / "aux epochs 3, 12 et 40"
    if len(numeros) == 1:
        return f"à l'epoch {numeros[0]}"
    if len(numeros) == 2:
        return f"à l'epoch {numeros[0]} et à l'epoch {numeros[1]}"
    return "aux epochs " + ", ".join(str(n) for n in numeros[:-1]) + f" et {numeros[-1]}"


def affiche_tableau(sujet_id, res, rms_ref, meilleure):
    # Tout le résultat du sujet à l'écran : le détail par epoch, puis les trois synthèses
    eps = sorted(int(e) for e in res)
    accs = [res[str(e)]["acc"] for e in eps]
    rmses = [res[str(e)]["rmse"] for e in eps]

    print(f"\n{sujet_id} — {len(eps)}/{N_EPOCHS} epochs — RMS de la référence ICLabel : {rms_ref:.2f} µV")
    print("  epoch |   accuracy    | RMSE (µV)")
    for e in eps:
        r = res[str(e)]
        marque = "   <- meilleure epoch de validation" if e == meilleure else ""
        print(f"   {e:4d} | {r['acc']:.2f} ± {r['std']:.2f}   |   {r['rmse']:6.2f}{marque}")

    #Ex æquo cherchés sur les valeurs telles qu'affichées, sinon la synthèse désignerait une
    #epoch que le tableau ne distingue pas
    acc_max = max(round(a, 2) for a in accs)
    rmse_min = min(round(r, 1) for r in rmses)
    print(f"\n  La meilleure accuracy ({acc_max:.2f}) se présente "
          f"{phrase_epochs([e for e, a in zip(eps, accs) if round(a, 2) == acc_max])}.")
    print(f"  Le RMSE minimal ({rmse_min:.1f} µV) se présente "
          f"{phrase_epochs([e for e, r in zip(eps, rmses) if round(r, 1) == rmse_min])}.")
    if str(meilleure) in res:
        b = res[str(meilleure)]
        print(f"  Au checkpoint retenu (epoch {meilleure}) : accuracy {b['acc']:.2f} ± {b['std']:.2f}, "
              f"RMSE {b['rmse']:.2f} µV, SNR {b['snr']:.2f} dB.")


for s in sujets:
    sujet_id = "S" + str(s).zfill(3)

    meilleure, rmse_val_min = meilleure_epoch_validation(sujet_id)
    if meilleure is None:
        print(f"{sujet_id} : aucun checkpoint dans {Modelsave}, ignoré", flush=True)
        continue

    f_brut = Pretraite / (sujet_id + "_Pre.fif")
    f_ref = Nettoye / sujet_id / Reference
    if not (f_brut.exists() and f_ref.exists()):
        print(f"{sujet_id} : fichiers manquants ({f_brut.name} / {f_ref.name}), ignoré", flush=True)
        continue

    epochs_brut = mne.read_epochs(f_brut, preload=True)
    data_brut = epochs_brut.get_data()
    ref = mne.read_epochs(f_ref, preload=True).get_data()
    rms_ref = float(np.sqrt(np.mean(ref ** 2)) * 1e6)
    print(f"{sujet_id} : meilleure epoch de validation = {meilleure} ({rmse_val_min:.2f} µV), "
          f"RMS de référence {rms_ref:.2f} µV", flush=True)

    #Reprise : les epochs déjà calculées sont relues au lieu d'être refaites, sauf si le
    #checkpoint qui les a produites a changé depuis
    fichier_json = Sortie_ART / sujet_id / "resultats_epochs.json"
    etat = json.loads(fichier_json.read_text(encoding="utf-8")) if fichier_json.exists() else {}
    res = {} if args.force else etat.get("epochs", {})

    perimes = []
    for e, v in res.items():
        c = lit_checkpoint(sujet_id, int(e))
        if c is None or v.get("modele") != c["modele"]:
            perimes.append(e)
    if perimes:
        print(f"{sujet_id} : {len(perimes)} epoch(s) calculées avec un checkpoint qui a changé "
              f"depuis (réentraînement) -> recalcul", flush=True)

    #La meilleure epoch d'abord : c'est celle qui compte, un run interrompu la fournit quand même
    a_traiter = [meilleure]
    if not args.meilleure_seule:
        a_traiter += [e for e in range(1, N_EPOCHS + 1) if e != meilleure]

    for ep in a_traiter:
        c = lit_checkpoint(sujet_id, ep)
        if c is None:
            continue
        deja = res.get(str(ep))
        if deja is not None and deja.get("modele") == c["modele"]:
            continue
        debut = time.time()

        data_clean = np.empty_like(data_brut)
        for i, essai in enumerate(data_brut):
            data_clean[i] = Utils.clean_epoch(essai, args.modele, sujet_id, ep)
        Utils._CACHE.clear()   # un checkpoint de 15 Mo par epoch : on ne les empile pas

        rmse = float(np.sqrt(np.mean((data_clean - ref) ** 2)) * 1e6)

        #Accuracy CSP + LDA sur les deux classes de mouvement (repos retiré)
        e = mne.EpochsArray(data_clean, epochs_brut.info, epochs_brut.events,
                            tmin=epochs_brut.tmin, event_id=epochs_brut.event_id)["gauche", "droite"]
        scores = cross_val_score(clf, prep(e.get_data()), e.events[:, 2], cv=cv)
        res[str(ep)] = {"rmse": rmse, "acc": float(scores.mean()), "std": float(scores.std()),
                        "snr": float(20 * np.log10(rms_ref / rmse)), "modele": c["modele"]}

        #Meilleure epoch : le signal débruité est conservé, pour CSP.py, Visualize.py et Evaluation.py
        if ep == meilleure:
            dossier = Sortie_ART / sujet_id
            dossier.mkdir(parents=True, exist_ok=True)
            mne.EpochsArray(data_clean, epochs_brut.info, epochs_brut.events,
                            tmin=epochs_brut.tmin, event_id=epochs_brut.event_id).save(
                dossier / f"ART_epoch{ep}.fif", overwrite=True)

        #Enregistré après chaque epoch : une exécution interrompue reprend sans rien recalculer
        etat.update({"meilleure_epoch": meilleure, "rms_ref": rms_ref, "epochs": res})
        fichier_json.parent.mkdir(parents=True, exist_ok=True)
        fichier_json.write_text(json.dumps(etat, indent=1), encoding="utf-8")
        print(f"  {sujet_id} epoch {ep}/{N_EPOCHS} : RMSE {rmse:.2f} µV, "
              f"acc {scores.mean():.2f} +/- {scores.std():.2f} ({time.time() - debut:.0f}s)", flush=True)

    affiche_tableau(sujet_id, res, rms_ref, meilleure)
