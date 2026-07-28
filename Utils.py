"""
Application des modeles de debruitage ART sur des essais (30 x 1024).
Adapte de ArtifactRemovalTransformer/utils.py, pour des donnees deja pretraitees.
"""

import json
import shutil
import subprocess
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
    # ART_ICLABEL : un checkpoint par sujet exclu (LOSO) et par epoch d'entrainement,
    # dans Model/ART_ICLABEL/modelsave/SXXX/Epoch_NY/checkpoint.pth.tar
    cle = (mode, sujet_id, epoch_num)
    if cle in _CACHE:
        return _CACHE[cle]
    if mode == "ART_ICLABEL":
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
    elif mode in ("ART", "ART_ICLABEL"):
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
        else:  # ART / ART_ICLABEL
            src = torch.FloatTensor(data).to(device).unsqueeze(0)
            batch = tf_data.Batch(src, src, 0)
            out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
            out = model.generator(out).permute(0, 2, 1)
            out = torch.cat((out, torch.zeros(1, 30, 1).to(device)), dim=2)
    return np.array(out.cpu()).astype(np.float64)[0]


def clean_epoch(epoch, mode, sujet_id=None, epoch_num=None):
    # z-score global -> debruitage -> retour a l'echelle d'origine
    std = np.std(epoch, axis=0)
    avg = np.average(epoch, axis=0)
    decoded = decode_data((epoch - avg) / std, mode, sujet_id, epoch_num)
    print(std)
    print(avg)
    return decoded * std + avg


def sauve_feuille_loso(fichier, sujet, mse_train, mse_eval, pertes_batches):
    # Ecrit/actualise une feuille (nom = sujet exclu) dans un classeur Excel :
    #   - epoch_train_mse / epoch_eval_mse : evolution globale (1 valeur par epoch)
    #   - batch_epoch_N : evolution locale des pertes batch au sein de l'epoch N
    colonnes = {"epoch_train_mse": mse_train, "epoch_eval_mse": mse_eval}
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


def maj_tableau_tex(json_path, tex_path, cle, ligne_tex, macro, ordre=None):
    # Met a jour une ligne d'un tableau LaTeX (Rapport_ART.pdf) : chaque ligne est
    # gardee dans un sidecar JSON (une entree par cle, ex. un Method d'Evaluation.py),
    # puis le fichier .tex regenere une macro \macro{...} (chargee HORS tableau,
    # \input a l'interieur d'un tabular casse l'alignement avec cette distribution LaTeX).
    data = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    data[cle] = ligne_tex
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    cles = [c for c in (ordre or data) if c in data]
    contenu = "\n".join(data[c] for c in cles)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(f"\\def\\{macro}{{{contenu}}}\n", encoding="utf-8", newline="\n")


def recompile_latex(tex_path):
    # Recompile un rapport LaTeX (2 passes, pour la table des matieres) sans jamais
    # faire planter le script appelant (pdflatex absent, PDF ouvert/verrouille, etc.)
    if shutil.which("pdflatex") is None:
        print(f"  (pdflatex introuvable : rapport {tex_path.name} non recompile)")
        return
    resultat = None
    for _ in range(2):
        resultat = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name],
                                  cwd=tex_path.parent, capture_output=True, text=True)
    if resultat is not None and resultat.returncode != 0:
        print(f"  (echec de compilation de {tex_path.name} - verifie qu'il n'est pas ouvert ailleurs)")
    else:
        print(f"  -> rapport mis a jour : {tex_path}")
