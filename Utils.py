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
    # ART_ICLABEL* : un checkpoint par sujet exclu (LOSO) et par epoch d'entrainement,
    # dans Model/<mode>/modelsave/SXXX/Epoch_NY/checkpoint.pth.tar. Le prefixe couvre les
    # variantes d'entrainement (ART_ICLABEL, ART_ICLABEL_Ameliore, ...) sans toucher au code.
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
    # NB : noms distincts des anciennes colonnes epoch_train_mse/epoch_eval_mse, qui
    # contenaient un MSE sur signal normalise (sans unite) : les deux ne se melangent pas.
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


def genere_recap(rapport_dir, sujets, colonnes, suffixe, macro, meilleur=None):
    # Construit un tableau recapitulatif (une ligne par sujet, une colonne par cle)
    # a partir des sidecars JSON deja ecrits par maj_tableau_tex (accuracy ou mse) ;
    # "-" si la cle n'a pas encore ete calculee pour ce sujet.
    # meilleur="max"/"min" : ajoute une ligne "Moyenne" par colonne, avec la meilleure
    # moyenne mise en gras (max pour l'accuracy, min pour le MSE).
    lignes = []
    valeurs_colonnes = [[] for _ in colonnes]
    for sujet in sujets:
        f = rapport_dir / "data" / f"{sujet}_{suffixe}.json"
        data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        cellules = [sujet]
        for i, cle in enumerate(colonnes):
            if cle in data:
                valeur = data[cle].split("&", 1)[1].rsplit("\\\\", 1)[0].strip()
                cellules.append(valeur)
                try:
                    valeurs_colonnes[i].append(float(valeur.split("$\\pm$")[0]))
                except ValueError:
                    pass
            else:
                cellules.append("-")
        lignes.append(" & ".join(cellules) + r" \\")

    if meilleur:
        fmt = "{:.2e}" if suffixe == "mse" else "{:.2f}"
        moyennes = [sum(v) / len(v) if v else None for v in valeurs_colonnes]
        ecarts = [np.std(v) if v else None for v in valeurs_colonnes]
        candidats = [m for m in moyennes if m is not None]
        top = (max if meilleur == "max" else min)(candidats) if candidats else None
        cellules = [r"\textbf{Moyenne}"]
        for m, e in zip(moyennes, ecarts):
            if m is None:
                cellules.append("-")
            else:
                texte = fmt.format(m) if suffixe == "mse" else f"{fmt.format(m)} $\\pm$ {fmt.format(e)}"
                cellules.append(rf"\textbf{{{texte}}}" if m == top else texte)
        lignes.append(r"\midrule " + " & ".join(cellules) + r" \\")

    tex_path = rapport_dir / "data" / f"recap_{suffixe}.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(f"\\def\\{macro}{{" + "\n".join(lignes) + "}\n", encoding="utf-8", newline="\n")


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
