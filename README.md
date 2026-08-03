# ART — Débruitage EEG & évaluation Motor-Imagery

Pipeline pour **débruiter** des signaux EEG d'imagerie motrice (base **EEGBCI**, imagerie
main gauche vs main droite, runs 4/8/12) avec plusieurs modèles de débruitage
(**ART, IC-U-Net, DuoCL, GCTNet, ICLabel**), puis **évaluer** l'effet du débruitage sur
le décodage gauche / droite (**CSP + LDA**). Les débruiteurs mono-canal sont entraînés sur
**EEGdenoiseNet**, ART est réentraîné en *leave-one-subject-out* sur EEGBCI.

Les résultats sont consolidés dans un rapport LaTeX, `Résultats/Rapport_ART.pdf`
(dossier hors dépôt).

---

## Le signal de référence : ICLabel

Dans tout le projet, **« ICLabel » désigne une seule chose** : l'ICA (infomax étendu)
décomposée sur le signal **continu à 64 canaux**, avant tout prétraitement, dont les
composantes sont triées par ICLabel — tout ce qui n'est classé ni *brain* ni *other* est
retiré. Décomposer sur le signal continu complet plutôt que sur des essais déjà découpés
et réduits à 30 canaux donne à l'ICA bien plus de données et de capteurs pour séparer les
sources, donc une identification des artefacts nettement plus fiable.

C'est ce signal qui sert **à la fois de cible d'entraînement à ART et de référence** pour
toutes les comparaisons (RMSE, SNR, accuracy).

---

## Installation

**1. Cloner le dépôt**

```bash
git clone https://github.com/AymanOuardani/ART.git
cd ART
```

**2. Télécharger les données et les poids** (trop volumineux pour git) depuis Google Drive,
puis **placer les deux dossiers à la racine du projet** :

- **`Databases/`** → https://drive.google.com/drive/folders/1rDK7eBiQbPzbMyNxVnF4uIIXrTW86EPC?usp=sharing
- **`Model/`** → https://drive.google.com/drive/folders/1LhZl7nkZQeIFZYLbYq37SGQGQ0No0rkP?usp=sharing

Arborescence attendue :

```
ART/
├── Databases/
│   ├── EEGBCI/            # signal continu brut, 1 fichier -raw.fif par sujet
│   ├── EEGBCI_ICLABEL/    # même signal nettoyé par ICLabel (produit par ICLABEL_Brut.py)
│   └── EEGdenoiseNet/     # pour Training_DuoGCT.py
├── Model/                 # architectures + poids .pth.tar
├── Output/                # créé automatiquement par les scripts
└── Résultats/             # rapport LaTeX (hors dépôt)
```

**3. Installer les dépendances** (Python 3.10–3.13)

```bash
pip install numpy scipy pandas scikit-learn mne mne-icalabel matplotlib tqdm einops openpyxl odfpy torch
```

> `openpyxl` est requis pour `resultats_LOSO.xlsx` (entraînement d'ART), `odfpy` pour
> `resultats_accuracy.ods` (`Training_DuoGCT.py`).

> **GPU** (fortement conseillé pour l'entraînement) : installer la build CUDA de PyTorch
> depuis <https://pytorch.org> à la place de `torch`.
> Vérifier : `python -c "import torch; print(torch.cuda.is_available())"`.

---

## Pipeline

```bash
# 1. Référence ICLabel : ICA sur le signal continu 64 canaux
python ICLABEL_Brut.py             # tous les sujets, reprend là où il s'est arrêté
                                   # -> Databases/EEGBCI_ICLABEL/S001-raw.fif

# 2. Prétraitement : 30 canaux, 256 Hz, blocs de 4 s
#    3 classes : gauche, droite, repos (tout le signal)
python Pretraitement.py 1            # brut     -> Output/Prétraité/S001_Pre.fif
python Pretraitement.py 1 --iclabel  # ICLabel  -> Output/Nettoyé/S001/ICLABEL.fif

# 3. Débruitage -> Output/Nettoyé/S001/<modele>.fif
python Clean_Model.py ART          # tous les sujets par défaut
python Clean_Model.py DuoCL 1-10   # sujets 1 à 10 seulement
python Clean_Model.py GCTNet --force

# 4. Évaluation CSP + LDA (gauche vs droite)
python Evaluation.py brut 1        # un sujet, et met à jour le rapport
python Evaluation.py ART           # tous les sujets, moyenne finale
```

Tous les signaux comparés partent du même `Prétraité` (3 classes) ; le repos est retiré
juste avant la classification, si bien que toutes les méthodes portent sur exactement les
mêmes essais.

> Le brut et sa version ICLabel passent par le **même** script de prétraitement, donc par
> exactement le même réordonnancement de canaux, le même rééchantillonnage, le même filtre
> et le même découpage. C'est indispensable puisqu'ils sont ensuite comparés échantillon par
> échantillon (RMSE, SNR) et forment la paire d'entraînement d'ART.

### Modèles disponibles

`Clean_Model.py` : `ART` · `ART_ICLABEL` · `ICUNet` · `ICUNet++` · `ICUNet_attn` · `DuoCL` ·
`GCTNet`. `Evaluation.py` accepte les mêmes, plus `brut` et `ICLABEL`.

| Modèle | Description |
|--------|--------------|
| `ART` | Transformer ART, poids d'origine (appelé `ART_Orig` dans le rapport) |
| `ART_ICLABEL` | ART réentraîné en LOSO contre ICLabel — nécessite un numéro d'epoch |
| `ICUNet` / `ICUNet++` / `ICUNet_attn` | familles IC-U-Net |
| `DuoCL` / `GCTNet` | débruiteurs mono-canal (EEGdenoiseNet) |
| `ICLABEL` | la référence (cf. plus haut) |

---

## Entraînement des débruiteurs

```bash
python Train_ART.py                          # ART neuf, LOSO, cible = ICLabel
python Training_DuoGCT.py DuoCL --device gpu
python Training_DuoGCT.py all --device cpu    # DuoCL + GCTNet
```

- **`Train_ART.py`** : pas d'argument, hyperparamètres fixes (60 epochs, batch 32, lr 0.01,
  Adam). Pour chaque sujet exclu, un modèle neuf est entraîné sur 86 des 108 autres sujets,
  validé sur les 22 restants, puis testé sur le sujet exclu au checkpoint de la meilleure
  epoch de validation. Produit un checkpoint par sujet **et par epoch** dans
  `Model/ART_ICLABEL/modelsave/SXXX/Epoch_NY/`, plus les courbes dans `resultats_LOSO.xlsx`.
  > **Attention** : aucune reprise. Le relancer réentraîne tout depuis le début et **écrase**
  > les checkpoints existants avec des poids différents.
- **`Training_DuoGCT.py`** (DuoCL/GCTNet, sur EEGdenoiseNet) : `DuoCL` · `GCTNet` · `all`.
  Options : `--device {auto,cpu,gpu}` · `--gpu N` · `--noise {EOG,EMG,Hybrid}` ·
  `--epochs N` · `--batch_size N`.

---

## Rapport

```bash
python Rapport_Sujets.py 1-16                           # chaîne complète, plage de sujets
python Rapport_Sujets.py 1-16 --modele ART_ICLABEL_v2   # autre dossier d'entraînement
python Rapport_Sujets.py 1-16 --sans-art                # sans les 60 epochs (rapide)
```

`Rapport_Sujets.py` enchaîne, pour chaque sujet, les scripts ci-dessous puis compile le PDF
une seule fois. Ils restent utilisables séparément :

| Script | Rôle |
|--------|------|
| `ART_Epochs.py` | applique les 60 checkpoints LOSO d'un sujet, mesure RMSE et accuracy par epoch, écrit le tableau et les deux figures |
| `mse_eval.py` | RMSE d'une méthode contre ICLabel, pour un sujet |
| `mse_graph.py` | courbe train/validation de l'entraînement LOSO d'un sujet |
| `Recap_ART.py` | tableau récapitulatif à la meilleure epoch de validation |

`ART_Epochs.py` écrit ses résultats dans `Résultats/data/SXXX_epochs.json` après **chaque**
epoch : une exécution interrompue reprend sans rien recalculer. Il enregistre aussi la date
du checkpoint utilisé, et **recalcule automatiquement** tout résultat issu d'un checkpoint
modifié depuis — indispensable après un réentraînement. Options utiles :
`--meilleure-seule` (seulement la meilleure epoch), `--rapport-seul` (régénère tableau et
figures sans recalcul), `--sans-latex` (plusieurs instances en parallèle).

---

## Inspection

```bash
python Visualize.py brut 1         # signal continu, brut et nettoyé ICLabel
python Visualize.py pretraite 1    # essais prétraités
python Visualize.py ART 1          # avant / après débruitage (2 fenêtres)
python CSP.py brut 1               # topographies des filtres CSP
python RMS.py 1                    # RMS de chaque méthode (idem RMSE.py, SNR.py, MAE.py)
```

---

## Les scripts

| Script | Rôle |
|--------|------|
| `ICLABEL_Brut.py` | ICA + tri ICLabel sur le signal continu 64 canaux → `Databases/EEGBCI_ICLABEL/` |
| `Pretraitement.py` | signal continu → format ART (30 canaux, 256 Hz, essais de 4 s) ; `--iclabel` pour la version nettoyée |
| `Clean_Model.py` | applique **un** modèle de débruitage → cache dans `Output/Nettoyé/` |
| `Train_ART.py` | entraîne un ART neuf en LOSO sur EEGBCI |
| `Training_DuoGCT.py` | entraîne DuoCL/GCTNet sur EEGdenoiseNet |
| `Evaluation.py` | décodage gauche/droite **CSP + LDA** |
| `Rapport_Sujets.py` | régénère les pages du rapport pour une plage de sujets |
| `ART_Epochs.py` · `Recap_ART.py` · `mse_eval.py` · `mse_graph.py` | briques du rapport |
| `CSP.py` · `Brut_TestCSP.py` | filtres CSP : topographies, balayage de `n_components` |
| `RMS.py` · `RMSE.py` · `SNR.py` · `MAE.py` | métriques d'un sujet, méthode par méthode |
| `Visualize.py` | inspection des signaux (brut / prétraité / débruité) |
| `Utils.py` | chargement d'un modèle, débruitage d'un essai, écriture du rapport |

> Ordre logique : `ICLABEL_Brut` → `Pretraitement` → `Train_ART` / `Training_DuoGCT` →
> `Clean_Model` → `Evaluation` / `Rapport_Sujets`.
