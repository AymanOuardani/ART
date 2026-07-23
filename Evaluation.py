"""
Evaluation CSP + LDA (Fig 6C), adaptee du code MNE (Billinger) a nos donnees.

Etape 3/3. Ce script NE fait QUE l'evaluation ; il consomme des donnees deja preparees :
  - Prétraité/S###-epo.fif         (sans traitement)     <- Pretreatement.py <sujets>
  - Nettoyé/S###/{modele}_{dataset}-epo.fif (debruite)   <- Clean.py <modele> <dataset>

En ligne de commande, on choisit UN modele a evaluer, identifie par son nom ET la
base sur laquelle il a ete entraine (ex. ART entraine sur la base d'origine vs ART
entraine sur EEGdenoiseNet). Le script :
  1. verifie que les essais debruites de ce modele existent dans Nettoyé/
     (sinon il indique la commande de debruitage a lancer d'abord) ;
  2. calcule l'accuracy CSP+LDA (10 runs holdout 80/20) par sujet ;
  3. AJOUTE / ACTUALISE la feuille du (modele, dataset) dans resultats_accuracy.ods,
     en conservant les feuilles deja presentes + "sans traitement" + "poolé".

    python Evaluation.py ART EEGdenoiseNet
    python Evaluation.py ART original
    python Evaluation.py DuoCL EEGdenoiseNet
    python Evaluation.py GCTNet EEGdenoiseNet
    python Evaluation.py ICUNet original

Le debruitage (application des modeles) est fait par Clean.py. Le registre des
modeles (MODEL_REGISTRY) y est defini et importe ici.
"""

import argparse

import numpy as np
import pandas as pd
import mne
from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

# Source unique du registre + conventions de nommage + dossiers (defini dans clean.py)
from Clean import (BASE, PRETRAITE, NETTOYE,
                   MODEL_REGISTRY, cache_key_of, label_of)

mne.set_log_level("ERROR")

OUT_ODS = BASE / "resultats_accuracy.ods"       # sortie : classeur des accuracies

# ------------------------------- Config -----------------------------------
SUBJECTS = range(1, 110)
SFREQ = 256
FMIN, FMAX = 7.0, 30.0                          # band-pass mu/beta avant CSP
# fenetre [1,2]s post-cue, decalee du retard du FIR causal 1-50 (~1.95 s) du pretraitement
CROP = slice(int(round(2.95 * SFREQ)), int(round(3.95 * SFREQ)))
N_ITER = 10                                     # 10 runs holdout


# --------------------------- Charger / preparer ----------------------------
def load_subject(subject, key):
    # key=None -> Prétraité (sans traitement) ; sinon Nettoyé/S###/{key}
    path = (PRETRAITE / f"S{subject:03d}-epo.fif" if key is None
            else NETTOYE / f"S{subject:03d}" / f"{key}-epo.fif")
    if not path.exists():
        return None, None
    ep = mne.read_epochs(path, verbose="ERROR")
    return ep.get_data(copy=True), ep.events[:, 2]   # X (n,30,1024), y (1=gauche,2=droite)


def prep(X):
    X = X - X.mean(axis=1, keepdims=True)            # reference moyenne
    X = filter_data(X, SFREQ, FMIN, FMAX, verbose="ERROR")
    return X[:, :, CROP]                             # fenetre [1,2]s


# ------------------------------ CSP + LDA ----------------------------------
def make_clf():
    return Pipeline([("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                     ("LDA", LinearDiscriminantAnalysis())])


def scores_10(X, y):
    # renvoie les 10 scores holdout 80/20 (un par iteration)
    cv = ShuffleSplit(N_ITER, test_size=0.2, random_state=42)
    return cross_val_score(make_clf(), X, y, cv=cv)  # array de 10 scores


_CACHE_PREP = {}


def pooled_prepped(key):
    # (X prepare, y, groupe=sujet) poole sur les 109 sujets, avec cache
    if key in _CACHE_PREP:
        return _CACHE_PREP[key]
    Xs, ys, gs = [], [], []
    for s in SUBJECTS:
        X, y = load_subject(s, key)
        if X is None:
            continue
        Xs.append(prep(X))
        ys.append(y)
        gs.append(np.full(len(y), s))
    res = (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs))
    _CACHE_PREP[key] = res
    return res


def pool_scores(key):
    X, y, _ = pooled_prepped(key)
    return scores_10(X, y)


# ------------------------- Feuille (DataFrame) par modele ------------------
def build_sheet(key, label):
    cols = [f"iter_{i + 1:02d}" for i in range(N_ITER)]
    rows, index = [], []
    for s in SUBJECTS:
        X, y = load_subject(s, key)
        if X is None:
            continue
        scores = scores_10(prep(X), y)
        rows.append(scores)
        index.append(f"S{s:03d}")
        print(f"  {label:24s}  S{s:03d} : moyenne {scores.mean():.2f}", flush=True)

    df = pd.DataFrame(rows, index=index, columns=cols)
    df.loc["moyenne"] = df.mean(axis=0)             # moyenne par iteration (sur sujets)
    df.loc["ecart-type"] = df.iloc[:-1].std(axis=0)  # ecart-type par iteration
    df.loc["moyenne finale"] = np.nan
    df.loc["ecart-type finale"] = np.nan
    df.loc["moyenne finale", cols[0]] = df.loc["moyenne"].mean()
    df.loc["ecart-type finale", cols[0]] = df.loc["ecart-type"].mean()
    return df


def upsert_pool(df, label, cache_key):
    # feuille poolee : 1 ligne par (modele, dataset), 10 iterations + moyenne/ecart-type.
    cols = [f"iter_{i + 1:02d}" for i in range(N_ITER)]
    if df is None:
        df = pd.DataFrame(columns=cols + ["moyenne", "ecart-type"])

    def row_for(key):
        sc = pool_scores(key)
        return list(sc) + [float(np.mean(sc)), float(np.std(sc))]

    if "sans traitement" not in df.index:
        df.loc["sans traitement"] = row_for(None)
        print(f"  poolé sans traitement    : {df.loc['sans traitement', 'moyenne']:.3f}", flush=True)
    df.loc[label] = row_for(cache_key)
    print(f"  poolé {label:18s} : {df.loc[label, 'moyenne']:.3f}", flush=True)
    return df


# ------------------------------ Fichier .ods -------------------------------
def load_existing_sheets():
    if OUT_ODS.exists():
        try:
            return pd.read_excel(OUT_ODS, sheet_name=None, engine="odf", index_col=0)
        except Exception:
            pass
    return {}


def write_sheets(sheets):
    # "sans traitement" en premier, "poolé" en dernier, feuilles modeles au milieu.
    middle = [k for k in sheets if k not in ("sans traitement", "poolé")]
    order = ([k for k in ["sans traitement"] if k in sheets]
             + middle
             + [k for k in ["poolé"] if k in sheets])
    with pd.ExcelWriter(OUT_ODS, engine="odf") as writer:
        for name in order:
            sheets[name].to_excel(writer, sheet_name=name)


def require_cleaned(cache_key, label, model, dataset):
    # Verifie que le debruitage a bien ete fait ; sinon, indique la commande a lancer.
    have = [s for s in SUBJECTS
            if (NETTOYE / f"S{s:03d}" / f"{cache_key}-epo.fif").exists()]
    if not have:
        raise SystemExit(
            f"ERREUR : aucun essai debruite trouve pour {label} dans Nettoyé/.\n"
            f"  Lancez d'abord le debruitage :\n"
            f"      python Clean.py {model} {dataset}")
    return have


# --------------------------------- Main ------------------------------------
def main():
    models = sorted({m for m, _ in MODEL_REGISTRY})
    datasets = sorted({d for _, d in MODEL_REGISTRY})
    ap = argparse.ArgumentParser(description="Evaluation CSP+LDA d'un modele de debruitage sur EEGBCI.")
    ap.add_argument("model", choices=models, help="nom du modele")
    ap.add_argument("dataset", choices=datasets,
                    help="base d'entrainement du modele (distingue les variantes, ex. ART)")
    opts = ap.parse_args()

    key = (opts.model, opts.dataset)
    if key not in MODEL_REGISTRY:
        avail = ", ".join(f"{m}/{d}" for m, d in MODEL_REGISTRY)
        raise SystemExit(f"ERREUR : couple (modele, dataset) inconnu : {opts.model}/{opts.dataset}\n"
                         f"  Couples disponibles : {avail}")
    cache_key = cache_key_of(opts.model, opts.dataset)  # nom de fichier de cache
    label = label_of(opts.model, opts.dataset)          # nom de feuille / ligne poolee
    print(f"Modele : {label} | evaluation seule")

    # 1) prerequis : le debruitage doit deja avoir ete fait par Pretreatement.py
    require_cleaned(cache_key, label, opts.model, opts.dataset)

    # 2) feuilles existantes (on conserve tout ce qui est deja calcule)
    sheets = load_existing_sheets()

    # 3) baseline "sans traitement" (calculee une seule fois, depuis Prétraité/)
    if "sans traitement" not in sheets:
        print("\n=== sans traitement ===", flush=True)
        sheets["sans traitement"] = build_sheet(None, "sans traitement")

    # 4) feuille du modele choisi (recalculee a chaque lancement)
    print(f"\n=== {label} ===", flush=True)
    sheets[label] = build_sheet(cache_key, label)

    # 5) feuille poolee (upsert de la baseline + du modele)
    print("\n=== poolé ===", flush=True)
    sheets["poolé"] = upsert_pool(sheets.get("poolé"), label, cache_key)

    # 6) ecriture
    write_sheets(sheets)
    print(f"\nFichier ecrit : {OUT_ODS}  (feuilles : {', '.join(sheets)})")


if __name__ == "__main__":
    main()
