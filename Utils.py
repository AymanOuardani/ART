"""
Fonctions partagées par les autres scripts — ce fichier ne se lance pas directement.

Chargement d'un modèle de débruitage et application à un essai (30 canaux x 1024 points) :
get_model, decode_data et clean_epoch, adaptés d'ArtifactRemovalTransformer/utils.py pour des
données déjà prétraitées. Plus sauve_feuille_loso, qui tient à jour le classeur des courbes
d'entraînement d'ART.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from Model import cumbersome_model2
from Model import UNet_family
from Model import UNet_attention
from Model import tf_model
from Model import tf_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(__file__).resolve().parent / "Model"
_CACHE = {}


def get_model(mode, sujet_id=None, epoch_num=None):
    # Construit et charge le modele une seule fois (mise en cache)
    # ART_ICLABEL* : un checkpoint par sujet exclu (LOSO) et par epoch d'entrainement,
    # dans Model/<mode>/modelsave/SXXX/Epoch_NY/checkpoint.pth.tar. Le prefixe couvre les
    # variantes d'entrainement (ART_ICLABEL, ART_ICLABEL_v2, ...) sans toucher au code.
    cle = (mode, sujet_id, epoch_num)
    if cle in _CACHE:
        return _CACHE[cle]
    if mode.startswith("ART_ICLABEL"):
        ckpt = MODEL_DIR / mode / "modelsave" / sujet_id / f"Epoch_N{epoch_num}" / "checkpoint.pth.tar"
    else:
        ckpt = MODEL_DIR / mode / "modelsave" / "checkpoint.pth.tar"
    state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]

    if mode == "ICUNet":
        model = cumbersome_model2.UNet1(n_channels=30, n_classes=30).to(device)
        model.load_state_dict(state, False)
    elif mode == "ICUNet++":
        model = UNet_family.NestedUNet3(num_classes=30).to(device)
        model.load_state_dict(state, False)
    elif mode == "ICUNet_attn":
        model = UNet_attention.UNetpp3_Transformer(num_classes=30).to(device)
        model.load_state_dict(state, False)
    elif mode == "ART" or mode.startswith("ART_ICLABEL"):
        model = tf_model.make_model(30, 30, N=2).to(device)
        model.load_state_dict(state)
    else:
        raise ValueError(f"modele inconnu : {mode}")

    model.eval()
    _CACHE[cle] = model
    return model


def decode_data(data, mode, sujet_id=None, epoch_num=None):
    # Debruite un bloc (30, 1024) deja normalise avec le modele demande
    model = get_model(mode, sujet_id, epoch_num)
    with torch.no_grad():
        if mode == "ICUNet":
            out = model(torch.Tensor(data[np.newaxis]).to(device))
        elif mode in ("ICUNet++", "ICUNet_attn"):
            _, _, out = model(torch.Tensor(data[np.newaxis]).to(device))
        else:  # ART / ART_ICLABEL*
            src = torch.FloatTensor(data).to(device).unsqueeze(0)
            batch = tf_data.Batch(src, src, 0)
            out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
            out = model.generator(out).permute(0, 2, 1)
            out = torch.cat((out, torch.zeros(1, 30, 1).to(device)), dim=2)
    return np.array(out.cpu()).astype(np.float64)[0]


def clean_epoch(epoch, mode, sujet_id=None, epoch_num=None):
    # z-score global (scalaire, tout le bloc canaux x temps) -> debruitage -> retour a l'echelle d'origine
    std = np.std(epoch)
    avg = np.average(epoch)
    decoded = decode_data((epoch - avg) / std, mode, sujet_id, epoch_num)
    return decoded * std + avg


def sauve_feuille_loso(fichier, sujet, rmse_train, rmse_val, pertes_batches):
    # Ecrit/actualise une feuille (nom = sujet exclu) dans un classeur Excel :
    #   - epoch_train_rmse / epoch_val_rmse : evolution globale en µV (1 valeur par epoch)
    #   - batch_epoch_N : evolution locale des pertes batch au sein de l'epoch N
    colonnes = {"epoch_train_rmse": rmse_train, "epoch_val_rmse": rmse_val}
    for i, pertes in enumerate(pertes_batches):
        colonnes[f"batch_epoch_{i + 1}"] = pertes
    feuille = pd.DataFrame({nom: pd.Series(vals) for nom, vals in colonnes.items()})

    feuilles = {}
    if fichier.exists():
        try:
            feuilles = pd.read_excel(fichier, sheet_name=None, engine="openpyxl")
        except Exception:
            feuilles = {}
    feuilles[sujet] = feuille
    with pd.ExcelWriter(fichier, engine="openpyxl") as writer:
        for nom, df in feuilles.items():
            df.to_excel(writer, sheet_name=str(nom)[:31], index=False)
