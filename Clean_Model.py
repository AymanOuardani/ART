"""
Applique un modèle de débruitage aux essais prétraités d'un sujet, et enregistre le résultat
dans Output/Nettoyé/SXXX/. Le fichier existant est toujours remplacé.

  python Clean_Model.py ART_Orig            les 109 sujets
  python Clean_Model.py ART_Orig 1          sujet 1
  python Clean_Model.py DuoCL 5
  python Clean_Model.py ART_Local 1 40      sujet 1, checkpoint de l'epoch 40
  python Clean_Model.py ART_Local tout 40   les 109 sujets à l'epoch 40

"""

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
Output = pl.Path(r"C:\Users\aymen\Desktop\ART\Output")
Model_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Model")
Pretraite = Output / "Prétraité"
Nettoye = Output / "Nettoyé"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Win = 512   # fenêtre des modèles mono-canal (EEGdenoiseNet)

#Modèles disponibles. DuoCL et GCTNet sont mono-canal et s'appliquent canal par canal,
#les autres traitent les 30 canaux d'un coup.
#ART_Local a un checkpoint par sujet et par epoch, d'où le numéro d'epoch.
Modeles = ["ART_Orig", "ART_Local", "ICUNet", "ICUNet++", "ICUNet_attn", "DuoCL", "GCTNet"]
Mono_canal = ["DuoCL", "GCTNet"]

#Ligne de commande
parser = ap.ArgumentParser(description="Débruitage EEGBCI")
parser.add_argument("Modele", choices=Modeles, help="modèle de débruitage")
parser.add_argument("Sujet", nargs="?", default="tout",
                    help="numéro du sujet, ou tout pour les 109 (défaut : tout)")
parser.add_argument("Epoch", type=int, nargs="?", default=None,
                    help="numéro d'epoch (requis pour ART_Local, ex. 40)")
args = parser.parse_args()
sujets = range(1, 110) if args.Sujet == "tout" else [int(args.Sujet)]

if args.Modele == "ART_Local" and args.Epoch is None:
    raise SystemExit("ERREUR : ART_Local nécessite un numéro d'epoch, ex. "
                     "python Clean_Model.py ART_Local 1 40")


def charge_modele_mono():
    #Réseau mono-canal, poids BEST de préférence
    dossier = Model_Dir / args.Modele / "modelsave"
    ckpt = dossier / "BEST_checkpoint.pth.tar"
    if not ckpt.exists():
        ckpt = dossier / "checkpoint.pth.tar"
    state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model = DuoCL(Win) if args.Modele == "DuoCL" else Generator(Win)
    model.load_state_dict(state)
    return model.to(device).eval()


def fenetres(total):
    #Débuts des fenêtres, la dernière ancrée à la fin
    debuts = list(range(0, total, Win))
    if debuts[-1] + Win > total:
        debuts[-1] = max(0, total - Win)
    return debuts


def debruite_epoch_mono(epoch, model):
    #Canal par canal, par fenêtres de 512 points z-scorées
    C, T = epoch.shape
    segs, meta = [], []
    for ch in range(C):
        for st in fenetres(T):
            seg = epoch[ch, st:st + Win]
            m, s = seg.mean(), seg.std()
            s = s if s > 1e-12 else 1.0
            segs.append(((seg - m) / s).astype(np.float32))
            meta.append((ch, st, m, s))
    x = torch.from_numpy(np.stack(segs)).to(device).unsqueeze(1)     # (N, 1, Win)
    with torch.no_grad():
        p = model(x).view(x.shape[0], -1)
    p = p.cpu().numpy()
    sortie = np.array(epoch, copy=True)
    for k, (ch, st, m, s) in enumerate(meta):
        sortie[ch, st:st + Win] = p[k] * s + m
    return sortie


#Modèle mono-canal si besoin, les autres passent par Utils
mono = args.Modele in Mono_canal
model_obj = charge_modele_mono() if mono else None

for s in sujets:
    sujet_id = "S" + str(s).zfill(3)

    #Lecture des essais prétraités du sujet
    fif_initial = Pretraite / (sujet_id + "_Pre.fif")
    if not fif_initial.exists():
        print(f"{sujet_id} : fichier introuvable ({fif_initial})")
        continue
    epochs = mne.read_epochs(fif_initial, preload=True)

    #Débruitage essai par essai
    data = epochs.get_data()
    data_clean = np.empty_like(data)
    for i, epoch in enumerate(data):
        if mono:
            data_clean[i] = debruite_epoch_mono(epoch, model_obj)
        else:
            data_clean[i] = Utils.clean_epoch(epoch, args.Modele, sujet_id, args.Epoch)

    #Sauvegarde, la sortie est toujours réécrite
    out = Nettoye / sujet_id / (args.Modele + ".fif")
    out.parent.mkdir(parents=True, exist_ok=True)
    mne.EpochsArray(data_clean, epochs.info, epochs.events,
                    tmin=epochs.tmin, event_id=epochs.event_id).save(out, overwrite=True)
    print(f"{sujet_id} : {len(data)} essais débruités -> {out}", flush=True)

    #ART_Local charge un modèle par sujet, inutile de tous les garder en mémoire
    if args.Modele == "ART_Local":
        Utils._cache.clear()
