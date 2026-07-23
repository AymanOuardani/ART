# ART — Débruitage EEG & évaluation Motor-Imagery

Pipeline pour **débruiter** des signaux EEG d'imagerie motrice (base **EEGBCI**)
avec des modèles de débruitage (**ART, IC-U-Net, DuoCL, GCTNet**), puis **évaluer**
l'effet du débruitage sur le décodage gauche / droite (**CSP + LDA**).
Les débruiteurs sont entraînés sur la base **EEGdenoiseNet**.

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
├── Databases/        # téléchargé  (EEGdenoiseNet + EEGBCI)
├── Model/            # téléchargé  (architectures + poids .pth.tar)
├── Output/           # créé automatiquement
├── Pretreatement.py  Clean.py  Evaluation.py  Training.py  Visualize.py  Utils.py
```

**3. Installer les dépendances** (Python 3.10–3.13)

```bash
pip install numpy scipy pandas scikit-learn mne matplotlib tqdm einops odfpy torch
```

> **GPU** (fortement conseillé pour l'entraînement) : installer la build CUDA de
> PyTorch depuis <https://pytorch.org> à la place de `torch`.
> Vérifier : `python -c "import torch; print(torch.cuda.is_available())"`.

---

## Pipeline (arguments **positionnels**)

```bash
# 1. Prétraitement EEGBCI  ->  Output/Prétraité/   (1re fois : télécharge les EDF)
python Pretreatement.py 1-109

# 2. Débruitage par un modèle  ->  Output/Nettoyé/
python Clean.py ART EEGdenoiseNet

# 3. Évaluation CSP+LDA  ->  resultats_accuracy.ods
python Evaluation.py ART EEGdenoiseNet
```

- **Modèles** : `ART` · `ICUNet` · `ICUNet++` · `ICUNet_attn` · `DuoCL` · `GCTNet`
- **Bases** : `original` (poids fournis) · `EEGdenoiseNet` (réentraînés)

---

## Entraînement des débruiteurs (sur EEGdenoiseNet)

```bash
python Training.py ART --device gpu       # modèle : ART | DuoCL | GCTNet | all
python Training.py all --device cpu       # device : auto | cpu | gpu
```

Produit les poids dans `Model/<modèle>/modelsave/` (`BEST_checkpoint.pth.tar`).

---

## Visualisation

```bash
python Visualize.py brut 1                 # étape : brut | pretraite | nettoye
python Visualize.py nettoye 1 ART
```

---

## Les scripts

| Script | Rôle |
|--------|------|
| `Pretreatement.py` | EEGBCI brut → format ART (30 canaux, 256 Hz, essais 4 s) |
| `Clean.py` | applique **un** modèle de débruitage → cache dans `Output/Nettoyé/` |
| `Evaluation.py` | décodage gauche/droite **CSP + LDA** → `resultats_accuracy.ods` |
| `Training.py` | entraîne les débruiteurs (ART/DuoCL/GCTNet) sur EEGdenoiseNet |
| `Visualize.py` | inspection des signaux (brut / prétraité / nettoyé) |
| `Utils.py` | chargement d'un modèle + débruitage d'un essai (multi-canal) |

> Ordre logique : `Training` (une fois, pour les poids) puis
> `Pretreatement → Clean → Evaluation` pour chaque modèle à tester.
