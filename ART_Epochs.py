import argparse as ap
import json
import pathlib as pl
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # tâche de fond : aucune fenêtre, on écrit directement les PNG
import matplotlib.pyplot as plt
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
Rapport_Tex = pl.Path(r"C:\Users\aymen\Desktop\ART\Résultats\Rapport_ART.tex")
Images_Dir = Rapport_Tex.parent / "Images"
Data_Dir = Rapport_Tex.parent / "data"

#Signal de référence : l'ICA ICLabel appliquée au signal continu à 64 canaux (ICLABEL_Brut.py).
#C'est la cible d'entraînement d'ART (cf. Train_ART.py) et le seul ICLabel du rapport.
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
parser = ap.ArgumentParser(description="ART (LOSO) epoch par epoch : RMSE et accuracy, pour le rapport")
parser.add_argument("Sujets", nargs="?", default="1-109", help="ex. 1-16 ou 1,2,5 (défaut : 1-109)")
parser.add_argument("--force", action="store_true", help="recalcule les epochs déjà enregistrées")
parser.add_argument("--sans-latex", action="store_true",
                    help="n'appelle pas pdflatex (à utiliser quand plusieurs instances tournent en parallèle)")
parser.add_argument("--rapport-seul", action="store_true",
                    help="régénère seulement le tableau et les figures depuis les résultats déjà calculés")
parser.add_argument("--meilleure-seule", action="store_true",
                    help="ne traite que la meilleure epoch de validation (signal .fif + métriques), sans les 59 autres")
parser.add_argument("--modele", default=Modele_Defaut,
                    help=f"dossier des checkpoints LOSO dans Model/ (défaut : {Modele_Defaut})")
args = parser.parse_args()

Modele = Model_Dir / args.modele
Fichier_Excel = Modele / "resultats_LOSO.xlsx"     # courbes train/validation de Train_ART.py

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


def horodatage_checkpoint(sujet_id, ep):
    # Date du checkpoint ayant servi au calcul. Train_ART.py n'a pas de reprise : le relancer
    # réentraîne tout depuis le début et réécrit les checkpoints avec des poids différents,
    # sans rien signaler. On garde cette date à côté de chaque résultat pour ne jamais
    # réutiliser une valeur produite par un modèle qui n'existe plus.
    f = Modele / "modelsave" / sujet_id / f"Epoch_N{ep}" / "checkpoint.pth.tar"
    return round(f.stat().st_mtime, 3) if f.exists() else None


def phrase_epochs(numeros):
    # "à l'epoch 8" / "à l'epoch 49 et à l'epoch 58" / "aux epochs 3, 12 et 40"
    if len(numeros) == 1:
        return f"à l'epoch {numeros[0]}"
    if len(numeros) == 2:
        return f"à l'epoch {numeros[0]} et à l'epoch {numeros[1]}"
    return "aux epochs " + ", ".join(str(n) for n in numeros[:-1]) + f" et {numeros[-1]}"


def courbe(eps, valeurs, optimums, ylabel, chemin):
    # Même présentation que mse_graph.py : la courbe, et l'optimum (ou les ex æquo) en rouge
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(eps, valeurs)
    ymin, ymax = ax.get_ylim()
    for ep in optimums:
        v = valeurs[eps.index(ep)]
        ax.scatter([ep], [v], color="red", zorder=5)
        ax.vlines(ep, ymin, v, color="red", linestyle="--", linewidth=1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Images_Dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def ecrit_rapport(sujet_id, res, rms_ref):
    # Tableau, phrases et figures du sujet, à partir des epochs déjà calculées
    eps = sorted(int(e) for e in res)
    accs = [res[str(e)]["acc"] for e in eps]
    rmses = [res[str(e)]["rmse"] for e in eps]
    Data_Dir.mkdir(parents=True, exist_ok=True)

    lignes = [f"{e} & {res[str(e)]['acc']:.2f} $\\pm$ {res[str(e)]['std']:.2f} & {res[str(e)]['rmse']:.2f} \\\\"
              for e in eps]
    (Data_Dir / f"{sujet_id}_epochs.tex").write_text(
        "\\def\\epochrows{\n" + "\n".join(lignes) + "}\n", encoding="utf-8", newline="\n")

    (Data_Dir / f"{sujet_id}_rms_ref_sentence.tex").write_text(
        "\\def\\rmsrefsentence{Le RMS de référence (signal nettoyé par ICLabel) pour "
        f"{sujet_id} est de {rms_ref:.1f} µV.}}\n", encoding="utf-8", newline="\n")

    #Ex æquo cherchés sur les valeurs telles qu'elles sont affichées (2 décimales pour l'accuracy,
    #1 pour le RMSE), sinon la phrase désignerait une epoch que le tableau ne distingue pas.
    acc_max = max(round(a, 2) for a in accs)
    rmse_min = min(round(r, 1) for r in rmses)
    best_acc = [e for e, a in zip(eps, accs) if round(a, 2) == acc_max]
    best_rmse = [e for e, r in zip(eps, rmses) if round(r, 1) == rmse_min]
    (Data_Dir / f"{sujet_id}_epochs_sentences.tex").write_text(
        f"\\def\\bestaccsentence{{La meilleure accuracy ({acc_max:.2f}) se présente "
        f"{phrase_epochs(best_acc)}.}}\n"
        f"\\def\\bestrmsesentence{{Le RMSE minimal ({rmse_min:.1f} µV) se présente "
        f"{phrase_epochs(best_rmse)}.}}\n", encoding="utf-8", newline="\n")

    courbe(eps, accs, best_acc, "Accuracy", Images_Dir / f"{sujet_id}_epochs_accuracy.png")
    courbe(eps, rmses, best_rmse, "RMSE (µV)", Images_Dir / f"{sujet_id}_epochs_rmse.png")


if not Fichier_Excel.exists():
    raise SystemExit(f"ERREUR : fichier introuvable : {Fichier_Excel}\n  (lance d'abord Train_ART.py)")
feuilles = pd.read_excel(Fichier_Excel, sheet_name=None, engine="openpyxl")

for s in sujets:
    sujet_id = "S" + str(s).zfill(3)

    #L'entraînement LOSO du sujet doit être terminé : c'est lui qui donne la meilleure epoch
    if sujet_id not in feuilles or "epoch_val_rmse" not in feuilles[sujet_id]:
        print(f"{sujet_id} : aucune donnée d'entraînement dans {Fichier_Excel.name}, ignoré", flush=True)
        continue
    val = feuilles[sujet_id]["epoch_val_rmse"].dropna().values
    if len(val) < N_EPOCHS:
        print(f"{sujet_id} : entraînement incomplet ({len(val)}/{N_EPOCHS} epochs), ignoré", flush=True)
        continue
    meilleure = int(np.argmin(val)) + 1

    f_brut = Pretraite / (sujet_id + "_Pre.fif")
    f_ref = Nettoye / sujet_id / Reference
    if not (f_brut.exists() and f_ref.exists()):
        print(f"{sujet_id} : fichiers manquants ({f_brut.name} / {f_ref.name}), ignoré", flush=True)
        continue

    #Reprise : les epochs déjà calculées sont relues au lieu d'être refaites
    fichier_json = Data_Dir / f"{sujet_id}_epochs.json"
    etat = json.loads(fichier_json.read_text(encoding="utf-8")) if fichier_json.exists() else {}
    res = {} if args.force else etat.get("epochs", {})

    #Régénération du tableau et des figures seuls, sans repasser par le débruitage
    if args.rapport_seul:
        if not res:
            print(f"{sujet_id} : aucun résultat enregistré, ignoré", flush=True)
            continue
        ecrit_rapport(sujet_id, res, etat["rms_ref"])
        print(f"{sujet_id} : {len(res)}/{N_EPOCHS} epochs, tableau et figures écrits", flush=True)
        continue

    epochs_brut = mne.read_epochs(f_brut, preload=True)
    data_brut = epochs_brut.get_data()
    ref = mne.read_epochs(f_ref, preload=True).get_data()
    rms_ref = float(np.sqrt(np.mean(ref ** 2)) * 1e6)
    print(f"{sujet_id} : meilleure epoch de validation = {meilleure} ({val.min():.2f} µV), "
          f"RMS de référence {rms_ref:.2f} µV", flush=True)

    #Résultats issus d'un modèle qui a changé depuis : ils seront recalculés, pas réutilisés
    perimes = [e for e, v in res.items() if v.get("modele") != horodatage_checkpoint(sujet_id, int(e))]
    if perimes:
        print(f"{sujet_id} : {len(perimes)} epoch(s) calculées avec un checkpoint qui a changé "
              f"depuis (réentraînement) -> recalcul", flush=True)

    #La meilleure epoch d'abord : c'est celle du récapitulatif, un run interrompu la fournit quand même
    a_traiter = [meilleure]
    if not args.meilleure_seule:
        a_traiter += [e for e in range(1, N_EPOCHS + 1) if e != meilleure]
    for ep in a_traiter:
        horodatage = horodatage_checkpoint(sujet_id, ep)
        deja = res.get(str(ep))
        #On saute une epoch seulement si son résultat vient bien du checkpoint actuel ; et pour
        #la meilleure epoch, il faut en plus que le bloc de métriques porte sur cette epoch-là
        #(après réentraînement, la meilleure epoch de validation change).
        if deja is not None and deja.get("modele") == horodatage \
                and (ep != meilleure or etat.get("meilleure", {}).get("epoch") == meilleure):
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
                        "modele": horodatage}

        #Meilleure epoch : signal conservé (CSP.py, vérifications) et métriques du récapitulatif
        if ep == meilleure:
            dossier = Sortie_ART / sujet_id
            dossier.mkdir(parents=True, exist_ok=True)
            mne.EpochsArray(data_clean, epochs_brut.info, epochs_brut.events,
                            tmin=epochs_brut.tmin, event_id=epochs_brut.event_id).save(
                dossier / f"ART_epoch{ep}.fif", overwrite=True)
            etat["meilleure"] = {"epoch": ep, "rmse": rmse,
                                 "snr": float(20 * np.log10(rms_ref / rmse)),
                                 "acc": res[str(ep)]["acc"], "std": res[str(ep)]["std"]}

        etat.update({"meilleure_epoch": meilleure, "rms_ref": rms_ref, "epochs": res})
        fichier_json.write_text(json.dumps(etat, indent=1), encoding="utf-8")
        print(f"  {sujet_id} epoch {ep}/{N_EPOCHS} : RMSE {rmse:.2f} µV, "
              f"acc {scores.mean():.2f} +/- {scores.std():.2f} ({time.time() - debut:.0f}s)", flush=True)

    ecrit_rapport(sujet_id, res, rms_ref)
    print(f"{sujet_id} : {len(res)}/{N_EPOCHS} epochs, tableau et figures écrits", flush=True)
    if Rapport_Tex.exists() and not args.sans_latex:
        Utils.recompile_latex(Rapport_Tex)

print("TERMINÉ")
