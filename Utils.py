"""
Fonctions partagées par les autres scripts — ce fichier ne se lance pas directement.

Chargement d'un modèle de débruitage et application à un essai (30 canaux x 1024 points) :
get_model, decode_data et clean_epoch, adaptés d'ArtifactRemovalTransformer/utils.py pour des
données déjà prétraitées. Plus sauve_feuille, qui tient à jour le classeur des courbes
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
Model_Dir = Path(__file__).resolve().parent / "Model"
_cache = {}


def get_model(mode, sujet_id=None, epoch_num=None):
    #Charge le modele une seule fois, puis le garde en cache.
    #ART_Local a un checkpoint par sujet et par epoch, les autres un seul
    cle = (mode, sujet_id, epoch_num)
    if cle in _cache:
        return _cache[cle]
    if mode.startswith("ART_Local"):
        ckpt = Model_Dir / mode / "modelsave" / sujet_id / f"Epoch_N{epoch_num}" / "checkpoint.pth.tar"
    else:
        ckpt = Model_Dir / mode / "modelsave" / "checkpoint.pth.tar"
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
    elif mode == "ART_Orig" or mode.startswith("ART_Local"):
        model = tf_model.make_model(30, 30, N=2).to(device)
        model.load_state_dict(state)
    else:
        raise ValueError(f"modele inconnu : {mode}")

    model.eval()
    _cache[cle] = model
    return model


def decode_data(data, mode, sujet_id=None, epoch_num=None):
    #Debruite un bloc (30, 1024) deja normalise
    model = get_model(mode, sujet_id, epoch_num)
    with torch.no_grad():
        if mode == "ICUNet":
            out = model(torch.Tensor(data[np.newaxis]).to(device))
        elif mode in ("ICUNet++", "ICUNet_attn"):
            _, _, out = model(torch.Tensor(data[np.newaxis]).to(device))
        else:  # ART_Orig / ART_Local*
            src = torch.FloatTensor(data).to(device).unsqueeze(0)
            batch = tf_data.Batch(src, src, 0)
            out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
            out = model.generator(out).permute(0, 2, 1)
            out = torch.cat((out, torch.zeros(1, 30, 1).to(device)), dim=2)
    return np.array(out.cpu()).astype(np.float64)[0]


def clean_epoch(epoch, mode, sujet_id=None, epoch_num=None):
    #z-score du bloc entier, debruitage, retour a l'echelle d'origine
    std = np.std(epoch)
    avg = np.average(epoch)
    decoded = decode_data((epoch - avg) / std, mode, sujet_id, epoch_num)
    return decoded * std + avg


def sauve_feuille(fichier, sujet, rmse_train, rmse_val, pertes_batches):
    #Une feuille par sujet : les deux courbes en µV, puis les pertes de chaque batch
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
