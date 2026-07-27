import argparse as ap
import pathlib as pl
import numpy as np
import torch
import mne as mne
import Utils
from Model.DuoCL import DuoCL
from Model.GCTNet import Generator

mne.set_log_level("ERROR")

#Chemins des fichiers
Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
Model_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Model")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

WIN = 512   # fenêtre des modèles mono-canal (EEGdenoiseNet)

#Modèles de débruitage disponibles (on nettoie toujours EEGBCI)
#  multi  : réseau 30->30 canaux, appliqué via Utils.clean_epoch
#  single : réseau mono-canal DuoCL/GCTNet, appliqué canal par canal
MODELES = {
    "ICUNet":      {"kind": "multi",  "mode": "ICUNet"},
    "ICUNet++":    {"kind": "multi",  "mode": "ICUNet++"},
    "ICUNet_attn": {"kind": "multi",  "mode": "ICUNet_attn"},
    "ART":         {"kind": "multi",  "mode": "ART"},
    "DuoCL":       {"kind": "single", "arch": "DuoCL",  "dossier": "DuoCL"},
    "GCTNet":      {"kind": "single", "arch": "GCTNet", "dossier": "GCTNet"},
}

#Ligne de commande
parser = ap.ArgumentParser(description="Débruitage EEGBCI")
parser.add_argument("Modele", choices=list(MODELES), help="modèle de débruitage")
parser.add_argument("Sujets", nargs="?", default="1-109", help="ex. 1-109 ou 1,2,5 (défaut : 1-109)")
parser.add_argument("--force", action="store_true", help="recalcule même si le cache existe déjà")
args = parser.parse_args()
entry = MODELES[args.Modele]

#"1-109" ou "1,2,5" -> liste de sujets
if "-" in args.Sujets:
    a, b = args.Sujets.split("-")
    sujets = list(range(int(a), int(b) + 1))
else:
    sujets = [int(s) for s in args.Sujets.split(",")]


def charge_modele_mono(entry):
    # Construit le réseau mono-canal et charge ses poids (BEST de préférence)
    dossier = Model_Dir / entry["dossier"] / "modelsave"
    ckpt = dossier / "BEST_checkpoint.pth.tar"
    if not ckpt.exists():
        ckpt = dossier / "checkpoint.pth.tar"
    state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model = DuoCL(WIN) if entry["arch"] == "DuoCL" else Generator(WIN)
    model.load_state_dict(state)
    return model.to(device).eval()


def fenetres(total):
    # Début de chaque fenêtre de WIN points couvrant [0, total[ (dernière ancrée à la fin)
    debuts = list(range(0, total, WIN))
    if debuts[-1] + WIN > total:
        debuts[-1] = max(0, total - WIN)
    return debuts


def debruite_epoch_mono(epoch, model):
    # epoch (30, T) -> (30, T), canal par canal, fenêtres de WIN=512 (z-score par fenêtre)
    C, T = epoch.shape
    segs, meta = [], []
    for ch in range(C):
        for st in fenetres(T):
            seg = epoch[ch, st:st + WIN]
            m, s = seg.mean(), seg.std()
            s = s if s > 1e-12 else 1.0
            segs.append(((seg - m) / s).astype(np.float32))
            meta.append((ch, st, m, s))
    x = torch.from_numpy(np.stack(segs)).to(device).unsqueeze(1)     # (N, 1, WIN)
    with torch.no_grad():
        p = model(x).view(x.shape[0], -1)
    p = p.cpu().numpy()
    sortie = np.array(epoch, copy=True)
    for k, (ch, st, m, s) in enumerate(meta):
        sortie[ch, st:st + WIN] = p[k] * s + m
    return sortie


#Chargement du modèle mono-canal si nécessaire (les modèles multi passent par Utils)
model_obj = charge_modele_mono(entry) if entry["kind"] == "single" else None

#Débruitage sujet par sujet, avec mise en cache dans Nettoyé/
n_ok = n_skip = n_absent = 0
for s in sujets:
    sujet_id = "S" + str(s).zfill(3)
    out = Nettoye / sujet_id / (args.Modele + ".fif")
    if out.exists() and not args.force:
        n_skip += 1
        continue
    fif_initial = Pretraite / (sujet_id + "-epo.fif")
    if not fif_initial.exists():
        n_absent += 1
        continue

    epochs = mne.read_epochs(fif_initial, preload=True)
    data = epochs.get_data()
    data_clean = np.empty_like(data)
    for i, epoch in enumerate(data):
        if entry["kind"] == "multi":
            data_clean[i] = Utils.clean_epoch(epoch, entry["mode"])
        else:
            data_clean[i] = debruite_epoch_mono(epoch, model_obj)

    epochs_clean = mne.EpochsArray(data_clean, epochs.info, epochs.events,
                                   tmin=epochs.tmin, event_id=epochs.event_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    epochs_clean.save(out, overwrite=True)
    n_ok += 1
    print("Sujet", s, "-> ", out.name)

print(f"\nTerminé : {n_ok} débruités, {n_skip} déjà en cache, {n_absent} sans prétraité.")
