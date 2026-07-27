"""
Application des modeles de debruitage ART sur des essais (30 x 1024).
Adapte de ArtifactRemovalTransformer/utils.py, pour des donnees deja pretraitees.
"""

from pathlib import Path
import numpy as np
import torch

from Model import cumbersome_model2
from Model import UNet_family
from Model import UNet_attention
from Model import tf_model
from Model import tf_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(__file__).resolve().parent / "Model"
_CACHE = {}


def get_model(mode):
    # Construit et charge le modele une seule fois (mise en cache)
    if mode in _CACHE:
        return _CACHE[mode]
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
    elif mode == "ART":
        model = tf_model.make_model(30, 30, N=2).to(device)
        model.load_state_dict(state)
    else:
        raise ValueError(f"modele inconnu : {mode}")

    model.eval()
    _CACHE[mode] = model
    return model


def decode_data(data, mode):
    # Debruite un bloc (30, 1024) deja normalise avec le modele demande
    model = get_model(mode)
    with torch.no_grad():
        if mode == "ICUNet":
            out = model(torch.Tensor(data[np.newaxis]).to(device))
        elif mode in ("ICUNet++", "ICUNet_attn"):
            _, _, out = model(torch.Tensor(data[np.newaxis]).to(device))
        else:  # ART
            src = torch.FloatTensor(data).to(device).unsqueeze(0)
            batch = tf_data.Batch(src, src, 0)
            out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
            out = model.generator(out).permute(0, 2, 1)
            out = torch.cat((out, torch.zeros(1, 30, 1).to(device)), dim=2)
    return np.array(out.cpu()).astype(np.float64)[0]


def clean_epoch(epoch, mode):
    # z-score global -> debruitage -> retour a l'echelle d'origine
    std = np.std(epoch, axis=0)
    avg = np.average(epoch, axis=0)
    decoded = decode_data((epoch - avg) / std, mode)
    print(std)
    print(avg)
    return decoded * std + avg
