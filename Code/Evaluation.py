"""
Evaluation CSP + LDA (Fig 6C), adaptee du code MNE (Billinger) a nos donnees.

En ligne de commande, on choisit UN modele a evaluer, identifie par son nom ET
la base sur laquelle il a ete entraine (ex. ART entraine sur la base d'origine
vs ART entraine sur EEGdenoiseNet). Le script :
  1. debruite a la volee les essais Prétraité/ avec ce modele (multi-canal via
     utils, ou mono-canal applique canal par canal en fenetres de 512), et met
     en cache le resultat dans Nettoyé/S###/{modele}_{dataset}-epo.fif ;
  2. calcule l'accuracy CSP+LDA (10 runs holdout 80/20) par sujet ;
  3. AJOUTE / ACTUALISE la feuille du (modele, dataset) dans resultats_accuracy.ods,
     en conservant les feuilles deja presentes + "sans traitement" + "poolé".

    python Evaluation.py --model ART    --dataset EEGdenoiseNet
    python Evaluation.py --model ART    --dataset original
    python Evaluation.py --model DuoCL  --dataset EEGdenoiseNet
    python Evaluation.py --model GCTNet --dataset EEGdenoiseNet
    python Evaluation.py --model ICUNet --dataset original

Poids attendus dans model/<dossier>/modelsave/ (voir MODEL_REGISTRY).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mne
from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

import utils
from model.DuoCL import DuoCL
from model.GCTNet import Generator
from model import tf_model, tf_data

mne.set_log_level("ERROR")

# ------------------------------- Config -----------------------------------
BASE = Path(__file__).resolve().parent
PRETRAITE = BASE / "Prétraité"
NETTOYE = BASE / "Nettoyé"
MODEL_DIR = BASE / "model"
OUT_ODS = BASE / "resultats_accuracy.ods"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SUBJECTS = range(1, 110)
SFREQ = 256
FMIN, FMAX = 7.0, 30.0                          # band-pass mu/beta avant CSP
# fenetre [1,2]s post-cue, decalee du retard du FIR causal 1-50 (~1.95 s) du pretraitement
CROP = slice(int(round(2.95 * SFREQ)), int(round(3.95 * SFREQ)))
N_ITER = 10                                     # 10 runs holdout
WIN = 512                                       # fenetre des modeles mono-canal (EEGdenoiseNet)

# Registre des modeles evaluables : (nom, dataset) -> configuration.
#   kind = "multi"  : reseau 30->30 canaux, applique via utils.clean_epoch(mode)
#   kind = "single" : reseau mono-canal (DuoCL/GCTNet), applique canal par canal
#   kind = "art512" : ART mono-canal make_model(1,1), applique canal par canal
# Pour ajouter un modele : ajouter une entree ici (+ les poids dans model/<folder>/).
MODEL_REGISTRY = {
    ("ICUNet", "original"):      {"kind": "multi",  "mode": "ICUNet"},
    ("ICUNet++", "original"):    {"kind": "multi",  "mode": "ICUNet++"},
    ("ICUNet_attn", "original"): {"kind": "multi",  "mode": "ICUNet_attn"},
    ("ART", "original"):         {"kind": "multi",  "mode": "ART"},
    ("ART", "EEGdenoiseNet"):    {"kind": "art512", "folder": "ART_EEGdenoiseNet"},
    ("DuoCL", "EEGdenoiseNet"):  {"kind": "single", "arch": "DuoCL",  "folder": "DuoCL"},
    ("GCTNet", "EEGdenoiseNet"): {"kind": "single", "arch": "GCTNet", "folder": "GCTNet"},
}


# --------------------------- Debruitage a la volee -------------------------
def art_forward(model, src):
    # src : (B, 1, T). Reproduit l'inference ART (encodeur=bruite, decodeur=bruite decale).
    batch = tf_data.Batch(src, src, 0)
    out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
    return model.generator(out).permute(0, 2, 1).squeeze(1)     # (B, T-1)


def build_single_model(entry):
    # Construit le reseau mono-canal et charge ses poids (BEST de preference).
    folder = MODEL_DIR / entry["folder"] / "modelsave"
    ckpt = folder / "BEST_checkpoint.pth.tar"
    if not ckpt.exists():
        ckpt = folder / "checkpoint.pth.tar"
    if not ckpt.exists():
        raise SystemExit(f"ERREUR : poids introuvables dans {folder}\n"
                         f"  (attendu BEST_checkpoint.pth.tar ou checkpoint.pth.tar). "
                         f"Entrainez d'abord ce modele.")
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)["state_dict"]
    if entry["kind"] == "art512":
        model = tf_model.make_model(1, 1, N=entry.get("layers", 2))
    elif entry["arch"] == "DuoCL":
        model = DuoCL(WIN)
    else:
        model = Generator(WIN)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def _windows(total):
    # Debut de chaque fenetre de WIN points couvrant [0, total[ (derniere ancree a la fin).
    starts = list(range(0, total, WIN))
    if starts[-1] + WIN > total:
        starts[-1] = max(0, total - WIN)
    return starts


def denoise_epoch_single(epoch, model, kind):
    # epoch (30, T) -> (30, T). Applique le modele mono-canal a chaque canal,
    # par fenetres de WIN=512 (z-score par fenetre, puis remise a l'echelle).
    C, T = epoch.shape
    starts = _windows(T)
    segs, meta = [], []
    for ch in range(C):
        for st in starts:
            seg = epoch[ch, st:st + WIN]
            m, s = seg.mean(), seg.std()
            s = s if s > 1e-12 else 1.0
            segs.append(((seg - m) / s).astype(np.float32))
            meta.append((ch, st, m, s))
    x = torch.from_numpy(np.stack(segs)).to(DEVICE).unsqueeze(1)     # (N, 1, WIN)
    with torch.no_grad():
        if kind == "art512":
            p = art_forward(model, x)                               # (N, WIN-1)
            p = torch.cat([p, torch.zeros(p.shape[0], 1, device=DEVICE)], dim=1)
        else:
            p = model(x).view(x.shape[0], -1)                       # (N, WIN)
    p = p.cpu().numpy()
    out = np.array(epoch, copy=True)
    for k, (ch, st, m, s) in enumerate(meta):
        out[ch, st:st + WIN] = p[k] * s + m
    return out


def ensure_cleaned(subject, entry, cache_key, model_obj):
    # Debruite les essais Prétraité/ du sujet et met en cache dans Nettoyé/ (si absent).
    out = NETTOYE / f"S{subject:03d}" / f"{cache_key}-epo.fif"
    if out.exists():
        return
    src = PRETRAITE / f"S{subject:03d}-epo.fif"
    if not src.exists():
        return
    ep = mne.read_epochs(src, verbose="ERROR")
    data = ep.get_data(copy=True)                    # (n, 30, 1024)
    cleaned = np.empty_like(data)
    for i, epoch in enumerate(data):
        if entry["kind"] == "multi":
            cleaned[i] = utils.clean_epoch(epoch, entry["mode"])
        else:
            cleaned[i] = denoise_epoch_single(epoch, model_obj, entry["kind"])
    out.parent.mkdir(parents=True, exist_ok=True)
    mne.EpochsArray(cleaned, ep.info, ep.events, tmin=ep.tmin,
                    event_id=ep.event_id, metadata=ep.metadata,
                    verbose="ERROR").save(out, overwrite=True)


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


# --------------------------------- Main ------------------------------------
def main():
    models = sorted({m for m, _ in MODEL_REGISTRY})
    datasets = sorted({d for _, d in MODEL_REGISTRY})
    ap = argparse.ArgumentParser(description="Evaluation CSP+LDA d'un modele de debruitage sur EEGBCI.")
    ap.add_argument("--model", required=True, choices=models, help="nom du modele")
    ap.add_argument("--dataset", required=True, choices=datasets,
                    help="base d'entrainement du modele (distingue les variantes, ex. ART)")
    opts = ap.parse_args()

    key = (opts.model, opts.dataset)
    if key not in MODEL_REGISTRY:
        avail = ", ".join(f"{m}/{d}" for m, d in MODEL_REGISTRY)
        raise SystemExit(f"ERREUR : couple (modele, dataset) inconnu : {opts.model}/{opts.dataset}\n"
                         f"  Couples disponibles : {avail}")
    entry = MODEL_REGISTRY[key]
    cache_key = f"{opts.model}_{opts.dataset}"          # nom de fichier de cache
    label = f"{opts.model} ({opts.dataset})"            # nom de feuille / ligne poolee
    print(f"Modele : {label} | kind : {entry['kind']} | device : {DEVICE}")

    # 1) modele mono-canal (le cas echeant) + debruitage/cache de tous les sujets
    model_obj = build_single_model(entry) if entry["kind"] in ("single", "art512") else None
    print(f"\n=== Debruitage {label} -> cache Nettoyé/ ===", flush=True)
    for s in SUBJECTS:
        ensure_cleaned(s, entry, cache_key, model_obj)

    # 2) feuilles existantes (on conserve tout ce qui est deja calcule)
    sheets = load_existing_sheets()

    # 3) baseline "sans traitement" (calculee une seule fois)
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
