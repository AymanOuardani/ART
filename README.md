# ART — Débruitage EEG & évaluation Motor-Imagery

Pipeline pour **débruiter** des signaux EEG d'imagerie motrice (base **EEGBCI**)
avec plusieurs modèles de débruitage (**ART, IC-U-Net, DuoCL, GCTNet, ICA/ICLabel**),
puis **évaluer** l'effet du débruitage sur le décodage gauche / droite (**CSP + LDA**).
Les débruiteurs neuronaux sont entraînés sur la base **EEGdenoiseNet**.

---

## Installation

**1. Cloner le dépôt**

```bash
git clone https://github.com/AymanOuardani/ART-ICUNET.git
cd ART-ICUNET
```

**2. Télécharger les données et les poids** (trop volumineux pour git) depuis
Google Drive, puis **placer les deux dossiers à la racine du projet** :

- **`Databases/`** → https://drive.google.com/drive/folders/1rDK7eBiQbPzbMyNxVnF4uIIXrTW86EPC?usp=sharing
- **`Model/`** → https://drive.google.com/drive/folders/1LhZl7nkZQeIFZYLbYq37SGQGQ0No0rkP?usp=sharing

Arborescence attendue :

```
ART-ICUNET/
├── Databases/          # téléchargé  (EEGdenoiseNet + EEGBCI)
├── Model/               # téléchargé  (architectures + poids .pth.tar)
├── Output/              # créé automatiquement par les scripts
├── Pretraitement.py  ICA.py  ICLABEL.py
├── Clean_Model.py  Training_Model.py  Evaluation.py  Visualize.py  Utils.py
```

**3. Installer les dépendances** (Python 3.10–3.13)

```bash
pip install numpy scipy pandas scikit-learn mne mne-icalabel matplotlib tqdm einops odfpy torch
```

> **GPU** (fortement conseillé pour l'entraînement) : installer la build CUDA de
> PyTorch depuis <https://pytorch.org> à la place de `torch`.
> Vérifier : `python -c "import torch; print(torch.cuda.is_available())"`.

---

## Pipeline

Tous les scripts nettoient toujours la base **EEGBCI** (109 sujets).

```bash
# 1. Prétraitement EEGBCI -> Output/Prétraité/  (1re fois : télécharge les EDF)
python Pretraitement.py 1
python Pretraitement.py 1 --total          # garde aussi le repos (T0) -> Prétraité_Total

# 2. Débruitage -> chaque sujet a un fichier par modèle dans Output/Nettoyé/S###/
python Clean_Model.py ART                  # -> Output/Nettoyé/S001/ART.fif
python ICA.py 1                            # ICA manuelle (sélection des composantes à l'œil)
python ICLABEL.py                          # ICA + ICLabel automatique, tous les sujets -> .../ICLABEL.fif

# 3. Évaluation CSP + LDA (affiche la précision moyenne)
python Evaluation.py ART
```

### Modèles disponibles (`Clean_Model.py` / `Evaluation.py`)

| Modèle | Description |
|--------|--------------|
| `ART` | Transformer ART, poids d'origine |
| `ART_EEGdenoiseNet` | ART réentraîné sur EEGdenoiseNet (mono-canal) |
| `ICUNet` / `ICUNet++` / `ICUNet_attn` | familles IC-U-Net |
| `DuoCL` / `GCTNet` | débruiteurs mono-canal (EEGdenoiseNet) |

```bash
python Clean_Model.py DuoCL 1-10           # sujets 1 à 10 seulement
python Clean_Model.py GCTNet --force       # recalcule même si déjà en cache
```

---

## Entraînement des débruiteurs

```bash
python Training_Model.py ART --device gpu
python Training_Model.py all --device cpu           # ART + DuoCL + GCTNet
python Training_Model.py ICLABEL                     # ART neuf, entraîné sur les paires ICLabel
```

- **Modèle** : `ART` · `DuoCL` · `GCTNet` · `ICLABEL` · `all`
- Options (ART/DuoCL/GCTNet, sur EEGdenoiseNet) : `--device {auto,cpu,gpu}` · `--gpu N` ·
  `--noise {EOG,EMG,Hybrid}` · `--epochs N` · `--batch_size N` · `--layers N`

Produit les poids dans `Model/<dossier>/modelsave/checkpoint.pth.tar` (+ `BEST_checkpoint.pth.tar`)
et l'évolution du MSE dans `resultats_accuracy.ods`.

---

## Visualisation

```bash
python Visualize.py brut 1                 # signal continu brut
python Visualize.py pretraite 1            # essais prétraités
python Visualize.py ART 1                  # avant / après débruitage (2 fenêtres)
python Visualize.py ICLABEL 1
```

Signaux disponibles : `brut` · `pretraite` · `ART` · `ICUNet` · `ICA` · `ICLABEL`.

---

## Les scripts

| Script | Rôle |
|--------|------|
| `Pretraitement.py` | EEGBCI brut → format ART (30 canaux, 256 Hz, essais 4 s) |
| `ICA.py` | ICA (infomax) + sélection manuelle des composantes à retirer |
| `ICLABEL.py` | ICA + tri automatique des composantes via ICLabel, tous les sujets |
| `Clean_Model.py` | applique **un** modèle de débruitage → cache dans `Output/Nettoyé/` |
| `Training_Model.py` | entraîne les débruiteurs (ART/DuoCL/GCTNet sur EEGdenoiseNet, ou ART sur ICLabel) |
| `Evaluation.py` | décodage gauche/droite **CSP + LDA**, affiche la précision moyenne |
| `Visualize.py` | inspection des signaux (brut / prétraité / débruité) |
| `Utils.py` | chargement d'un modèle + débruitage d'un essai (multi-canal) |

> Ordre logique : `Training_Model` (une fois, pour obtenir les poids) puis
> `Pretraitement → Clean_Model → Evaluation` pour chaque modèle à comparer.
