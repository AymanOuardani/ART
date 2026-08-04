"""
Décodage main gauche vs main droite par CSP + LDA, en validation croisée (10 tirages 80/20).

  python Evaluation.py brut 1        Sujet 1
  python Evaluation.py ART_Orig      les 109 sujets, avec la moyenne finale
  python Evaluation.py tout 1        toutes les méthodes, sujet 1
  python Evaluation.py tout          toutes les méthodes, les 109 sujets
  python Evaluation.py CSP tout      accuracy selon le nombre de composantes CSP
  python Evaluation.py CSP ART_Orig  le même balayage sur une seule méthode

Les résultats vont dans Output/Evaluation_Accuracies.xlsx, une feuille par méthode : les
10 runs de chaque sujet, puis les moyennes et écarts-types. Le balayage CSP écrit dans
Output/Evaluation_CSP.xlsx, une feuille par méthode, un sujet par ligne et un nombre de
composantes par colonne.
"""

import argparse as ap
import pathlib as pl
import numpy as np
import pandas as pd
import mne as mne
from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

mne.set_log_level("ERROR")

#Chemins des fichiers
Output = pl.Path(r"C:\Users\aymen\Desktop\ART\Output")
Pretraite = Output / "Prétraité"
Nettoye = Output / "Nettoyé"
Classeur = Output / "Evaluation_Accuracies.xlsx"
Classeur_CSP = Output / "Evaluation_CSP.xlsx"
classe1, classe2 = "gauche", "droite"   # imagerie main gauche / main droite (runs 4/8/12)

#Fichier de chaque signal. Tous viennent de Prétraité, repos compris.
fichiers = {"brut": None,
            "ART_Orig": "ART_Orig.fif",
            "ART_Local": "ART_Local.fif",
            "ICUNet": "ICUNet.fif",
            "ICUNet++": "ICUNet++.fif",
            "ICUNet_attn": "ICUNet_attn.fif",
            "DuoCL": "DuoCL.fif",
            "GCTNet": "GCTNet.fif",
            "ICLABEL": "ICLABEL.fif"}   # ICA ICLabel sur le raw 64 canaux (ICLABEL_Brut.py)

#Paramètres CSP + LDA
sfreq = 256
fmin, fmax = 7, 30
crop = slice(int(round(2.95 * sfreq)), int(round(3.95 * sfreq)))   # fenêtre [1,2]s (+ retard FIR ~1.95s)
N_iter = 10
n_defaut = 4                            # nombre de composantes de l'article
n_composantes = list(range(2, 15, 2))   # valeurs balayées en mode CSP

#Ligne de commande
parser = ap.ArgumentParser(description="Évaluation CSP + LDA")
parser.add_argument("Method", help="signal à évaluer (insensible à la casse) : CSP, tout, "
                                   + ", ".join(fichiers))
parser.add_argument("Sujet", nargs="?", default=None,
                    help="numéro du sujet (défaut : tous), ou le signal à balayer après CSP")
args = parser.parse_args()

#Insensible à la casse
correspondance = {nom.lower(): nom for nom in list(fichiers) + ["tout", "CSP"]}
if args.Method.lower() not in correspondance:
    raise SystemExit(f"ERREUR : signal inconnu '{args.Method}'. "
                     f"Choix possibles : CSP, tout, {', '.join(fichiers)}")
args.Method = correspondance[args.Method.lower()]

#En mode CSP le second argument désigne le signal, ailleurs c'est le numéro du sujet
balayage = args.Method == "CSP"
if balayage:
    cible = "tout" if args.Sujet is None else args.Sujet.lower()
    if cible not in correspondance:
        raise SystemExit(f"ERREUR : signal inconnu '{args.Sujet}'. "
                         f"Choix possibles : tout, {', '.join(fichiers)}")
    cible = correspondance[cible]
    methodes = list(fichiers) if cible == "tout" else [cible]
    sujets = range(1, 110)
else:
    methodes = list(fichiers) if args.Method == "tout" else [args.Method]
    sujets = range(1, 110) if args.Sujet is None else [int(args.Sujet)]

cv = ShuffleSplit(N_iter, test_size=0.2, random_state=42)


def charge_essais(methode, sujet_id):
    #Essais gauche/droite d'un sujet, référence moyenne retirée, 7-30 Hz, fenêtre [1,2]s
    if methode == "brut":
        fichier = Pretraite / (sujet_id + "_Pre.fif")
    else:
        fichier = Nettoye / sujet_id / fichiers[methode]
    if not fichier.exists():
        return None, None
    epochs = mne.read_epochs(fichier, preload=True)[classe1, classe2]   # retire le repos T0
    y = epochs.events[:, 2]                            # 1=gauche, 2=droite
    X = epochs.get_data()
    X = X - X.mean(axis=1, keepdims=True)
    X = filter_data(X, sfreq, fmin, fmax)
    return X[:, :, crop], y


def decodeur(methode, n):
    #ICLabel est rang-déficient après retrait de composantes, d'où la régularisation
    reg = "ledoit_wolf" if methode == "ICLABEL" else None
    return Pipeline([("CSP", CSP(n_components=n, reg=reg, log=True, norm_trace=False)),
                     ("LDA", LinearDiscriminantAnalysis())])


def ecris(classeur, feuilles):
    #Réécrit tout le classeur, appelé après chaque méthode pour ne rien perdre
    classeur.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(classeur, engine="openpyxl") as writer:
        for nom, df in feuilles.items():
            df.to_excel(writer, sheet_name=str(nom)[:31])


#Classeur relu une fois : les sujets déjà présents sont mis à jour, les autres conservés
sortie = Classeur_CSP if balayage else Classeur
feuilles = {}
if sortie.exists():
    try:
        feuilles = pd.read_excel(sortie, sheet_name=None, engine="openpyxl", index_col=0)
    except Exception:
        feuilles = {}

#Colonnes du balayage : la moyenne des 10 runs et son écart-type, pour chaque nombre de
#composantes, puis le meilleur nombre et l'accuracy qui va avec
resume = ["moyenne", "ecart-type", "moyenne finale", "ecart-type finale"]
colonnes_acc = [str(n) for n in n_composantes]
colonnes_csp = [c for n in colonnes_acc for c in (n, n + " ec")]
colonnes = (colonnes_csp + ["meilleur", "meilleure accuracy", "meilleure accuracy ec"] if balayage
            else [f"iter_{i + 1:02d}" for i in range(N_iter)])

for methode in methodes:

    lignes, noms, moyennes = [], [], []
    for s in sujets:
        sujet_id = "S" + str(s).zfill(3)
        X, y = charge_essais(methode, sujet_id)
        if X is None:
            print(f"  {methode:12s} {sujet_id} : fichier introuvable")
            continue

        if balayage:
            #Une accuracy par nombre de composantes, les essais n'étant chargés qu'une fois
            scores = [cross_val_score(decodeur(methode, n), X, y, cv=cv) for n in n_composantes]
            i = int(np.argmax([s.mean() for s in scores]))
            lignes.append([v for s in scores for v in (s.mean(), s.std())]
                          + [n_composantes[i], scores[i].mean(), scores[i].std()])
            print(f"  {methode:12s} {sujet_id} : meilleur {n_composantes[i]:2d} composantes "
                  f"-> {scores[i].mean():.2f} +/- {scores[i].std():.2f}", flush=True)
        else:
            scores = cross_val_score(decodeur(methode, n_defaut), X, y, cv=cv)   # 10 runs
            lignes.append(scores)
            moyennes.append(scores.mean())
            print(f"  {methode:12s} {sujet_id} : moyenne {scores.mean():.2f} "
                  f"+/- {scores.std():.2f}", flush=True)
        noms.append(sujet_id)

    if not lignes:
        continue

    #Une feuille par méthode, un sujet par ligne. Une feuille au mauvais format est refaite.
    feuille = feuilles.get(methode, pd.DataFrame(columns=colonnes))
    if list(feuille.columns) != colonnes:
        feuille = pd.DataFrame(columns=colonnes)
    feuille = feuille.drop(index=resume, errors="ignore")
    for nom, ligne in zip(noms, lignes):
        feuille.loc[nom] = ligne
    feuille = feuille.sort_index()

    if balayage:
        #Moyenne et dispersion inter-sujets de chaque colonne. Un nombre de composantes ne
        #se moyenne pas, d'où la case vide de la colonne meilleur.
        chiffrees = [c for c in colonnes if c != "meilleur"]
        feuille.loc["moyenne"] = feuille[chiffrees].mean().reindex(colonnes)
        feuille.loc["ecart-type"] = feuille[chiffrees].std(ddof=0).reindex(colonnes)
        best = int(np.argmax(feuille.loc["moyenne", colonnes_acc].values))
        print(f"  {methode:12s} moyenne maximale à {colonnes_acc[best]} composantes : "
              f"{feuille.loc['moyenne', colonnes_acc[best]]:.3f}\n")
    else:
        #Moyenne et écart-type par run, puis sur la moyenne de chaque sujet
        par_sujet = feuille.mean(axis=1)
        feuille.loc["moyenne"] = feuille.mean()
        feuille.loc["ecart-type"] = feuille.std(ddof=0)
        feuille.loc["moyenne finale"] = [par_sujet.mean()] + [np.nan] * (N_iter - 1)
        feuille.loc["ecart-type finale"] = [par_sujet.std(ddof=0)] + [np.nan] * (N_iter - 1)
        if len(moyennes) > 1:
            print(f"  {methode:12s} moyenne finale : {np.mean(moyennes):.3f} "
                  f"+/- {np.std(moyennes):.3f}\n")

    feuilles[methode] = feuille
    ecris(sortie, feuilles)

print(f"-> {sortie.name}")
