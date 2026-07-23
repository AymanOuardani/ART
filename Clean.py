"""
Etape 2/3 — Debruitage : Prétraité/ -> Nettoyé/.

Applique UN modele de debruitage a tous les essais prétraités, et met le resultat
en cache. Le modele est identifie par son nom ET la base sur laquelle il a ete
entraine (ex. ART d'origine vs ART reentraine sur EEGdenoiseNet).

  - multi-canal (ICUNet, ART d'origine) : bloc 30x1024 d'un coup, via utils
  - mono-canal  (DuoCL, GCTNet, ART reentraine) : canal par canal, fenetres de 512

    python Clean.py ART EEGdenoiseNet
    python Clean.py DuoCL EEGdenoiseNet 1-10
    python Clean.py ICUNet original

Sortie : Output/Nettoyé/S###/{modele}_{dataset}-epo.fif  (un fichier par sujet).
Le registre des modeles (MODEL_REGISTRY) est defini ici et importe par Evaluation.py.
Etape suivante : Evaluation.py (evaluation CSP+LDA).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import mne

import Utils
from Model.DuoCL import DuoCL
from Model.GCTNet import Generator
from Model import tf_model, tf_data

mne.set_log_level("ERROR")

# ------------------------------- Config -----------------------------------
BASE = Path(__file__).resolve().parent
PRETRAITE = BASE / "Output" / "Prétraité"     # entree : essais prétraités
NETTOYE = BASE / "Output" / "Nettoyé"         # sortie : essais debruites
MODEL_DIR = BASE / "Model"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WIN = 512                          # fenetre des modeles mono-canal (EEGdenoiseNet)

# Registre des modeles debruitables : (nom, dataset) -> configuration.
#   kind = "multi"  : reseau 30->30 canaux, applique via Utils.clean_epoch(mode)
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


def cache_key_of(model, dataset):
    # nom de fichier de cache dans Nettoyé/S###/
    return f"{model}_{dataset}"


def label_of(model, dataset):
    # nom lisible (feuille de resultats / affichage)
    return f"{model} ({dataset})"


def parse_subjects(text):
    # "1-109" ou "1,2,5" -> liste d'entiers
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


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
    # Renvoie "ok" (cree), "skip" (deja la) ou "absent" (pas de fichier prétraité).
    out = NETTOYE / f"S{subject:03d}" / f"{cache_key}-epo.fif"
    if out.exists():
        return "skip"
    src = PRETRAITE / f"S{subject:03d}-epo.fif"
    if not src.exists():
        return "absent"
    ep = mne.read_epochs(src, verbose="ERROR")
    data = ep.get_data(copy=True)                    # (n, 30, 1024)
    cleaned = np.empty_like(data)
    for i, epoch in enumerate(data):
        if entry["kind"] == "multi":
            cleaned[i] = Utils.clean_epoch(epoch, entry["mode"])
        else:
            cleaned[i] = denoise_epoch_single(epoch, model_obj, entry["kind"])
    out.parent.mkdir(parents=True, exist_ok=True)
    mne.EpochsArray(cleaned, ep.info, ep.events, tmin=ep.tmin,
                    event_id=ep.event_id, metadata=ep.metadata,
                    verbose="ERROR").save(out, overwrite=True)
    return "ok"


# --------------------------------- Main ------------------------------------
def main():
    models = sorted({m for m, _ in MODEL_REGISTRY})
    datasets = sorted({d for _, d in MODEL_REGISTRY})
    ap = argparse.ArgumentParser(description="Etape 2/3 : debruitage d'un modele (Prétraité -> Nettoyé).")
    ap.add_argument("model", choices=models, help="nom du modele")
    ap.add_argument("dataset", choices=datasets,
                    help="base d'entrainement du modele (distingue les variantes, ex. ART)")
    ap.add_argument("subjects", nargs="?", default="1-109", help="ex. 1-109 ou 1,2,5 (defaut : 1-109)")
    opts = ap.parse_args()

    key = (opts.model, opts.dataset)
    if key not in MODEL_REGISTRY:
        avail = ", ".join(f"{m}/{d}" for m, d in MODEL_REGISTRY)
        raise SystemExit(f"ERREUR : couple (modele, dataset) inconnu : {opts.model}/{opts.dataset}\n"
                         f"  Couples disponibles : {avail}")
    entry = MODEL_REGISTRY[key]
    cache_key = cache_key_of(opts.model, opts.dataset)
    label = label_of(opts.model, opts.dataset)
    print(f"Debruitage : {label} | kind : {entry['kind']} | device : {DEVICE}", flush=True)

    model_obj = build_single_model(entry) if entry["kind"] in ("single", "art512") else None
    n_ok = n_skip = n_absent = 0
    for s in parse_subjects(opts.subjects):
        status = ensure_cleaned(s, entry, cache_key, model_obj)
        n_ok += (status == "ok")
        n_skip += (status == "skip")
        n_absent += (status == "absent")
        if status == "ok":
            print(f"  S{s:03d} -> {cache_key}-epo.fif", flush=True)
    print(f"\nTermine : {n_ok} debruites, {n_skip} deja en cache, {n_absent} sans prétraité.")
    print(f"Sortie : Nettoyé/S###/{cache_key}-epo.fif")


if __name__ == "__main__":
    main()
