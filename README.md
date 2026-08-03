# ART — Débruitage EEG & évaluation Motor-Imagery

Pipeline pour **débruiter** des signaux EEG d'imagerie motrice (base **EEGBCI**, imagerie
main gauche vs main droite, runs 4/8/12) avec plusieurs modèles de débruitage
(**ART, IC-U-Net, DuoCL, GCTNet, ICLabel**), puis **évaluer** l'effet du débruitage sur le
décodage gauche / droite (**CSP + LDA**). Les débruiteurs mono-canal sont entraînés sur
**EEGdenoiseNet**, ART est réentraîné en *leave-one-subject-out* sur EEGBCI.

Tous les scripts affichent leurs résultats directement dans le terminal.

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
│   └── EEGdenoiseNet/     # pour l'entraînement de DuoCL et GCTNet
├── Model/                 # architectures + poids .pth.tar
└── Output/                # créé automatiquement par les scripts
```

**3. Installer les dépendances** (Python 3.10–3.13)

```bash
pip install numpy scipy pandas scikit-learn mne mne-icalabel matplotlib tqdm einops openpyxl odfpy torch
```

> `openpyxl` est requis pour `resultats_LOSO.xlsx` (entraînement d'ART), `odfpy` pour
> `resultats_accuracy.ods` (DuoCL / GCTNet).

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
python Evaluation.py brut 1        # un sujet
python Evaluation.py ART           # tous les sujets, moyenne finale
python Evaluation.py ART 4 1       # ART au checkpoint LOSO de l'epoch 1, sujet 4
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
| `ART` | Transformer ART, poids d'origine (appelé `ART_Orig` dans les comparaisons) |
| `ART_ICLABEL` | ART réentraîné en LOSO contre ICLabel — nécessite un numéro d'epoch |
| `ICUNet` / `ICUNet++` / `ICUNet_attn` | familles IC-U-Net |
| `DuoCL` / `GCTNet` | débruiteurs mono-canal (EEGdenoiseNet) |
| `ICLABEL` | la référence (cf. plus haut) |

---

## Entraînement

```bash
python Train_Model.py ART                    # LOSO sur EEGBCI, cible = ICLabel
python Train_Model.py DuoCL --device gpu     # sur EEGdenoiseNet
python Train_Model.py GCTNet
python Train_Model.py all                    # DuoCL puis GCTNet
```

Options communes : `--device {auto,cpu,gpu}` · `--gpu N` · `--epochs N` · `--batch_size N`.
Pour DuoCL et GCTNet : `--noise {EOG,EMG,Hybrid}`.
Pour ART : `--sujets S004,S007` pour ne relancer qu'une partie des folds, `--baseline` pour
n'afficher que la baseline identité sans entraîner.

**ART** : pour chaque sujet exclu, un modèle neuf est entraîné sur 86 des 108 autres sujets,
validé sur les 22 restants, puis testé sur le sujet exclu au checkpoint de la meilleure epoch
de validation. Le pas décroît en cosinus sur les 60 epochs. Chaque epoch affiche son RMSE
d'entraînement et de validation, en regard de la **baseline identité** — l'erreur qu'on
obtiendrait en recopiant simplement l'entrée bruitée, qui donne l'ordre de grandeur à battre.
Produit un checkpoint par sujet **et par epoch** dans `Model/ART_ICLABEL/modelsave/SXXX/Epoch_NY/`,
plus les courbes dans `resultats_LOSO.xlsx`.

> **Attention** : aucune reprise. Relancer réentraîne tout depuis le début et **écrase** les
> checkpoints existants avec des poids différents.

---

## Analyse

```bash
python Metrics.py tout 1           # RMS, RMSE, MAE et SNR de chaque méthode face à ICLabel
python Metrics.py SNR 1            # une seule métrique
python Metrics.py tout 1 --epoch 40    # ART pris à un checkpoint LOSO précis

python ART_Epochs.py 1-16          # applique les 60 checkpoints d'un sujet et affiche,
                                   # pour chaque epoch, RMSE contre ICLabel et accuracy
python mse_graph.py 4              # courbe train/validation de l'entraînement LOSO
```

`ART_Epochs.py` enregistre ses résultats dans `Output/ART/SXXX/resultats_epochs.json` après
**chaque** epoch : une exécution interrompue reprend sans rien recalculer. Il relève aussi la
date du checkpoint utilisé et **recalcule automatiquement** tout résultat issu d'un checkpoint
modifié depuis — indispensable après un réentraînement. La meilleure epoch de validation est
lue directement dans les checkpoints, et son signal débruité est conservé en `.fif` pour
`CSP.py`, `Visualize.py` et `Evaluation.py`. Options : `--meilleure-seule`, `--force`,
`--modele` pour pointer un autre dossier d'entraînement.

Le SNR se lit comme un rapport signal/bruit : 0 dB signifie que l'erreur a la même amplitude
que le signal de référence, 6 dB qu'elle en fait la moitié, 14 dB le cinquième.

---

## Inspection

```bash
python Visualize.py brut 1         # signal continu, brut et nettoyé ICLabel
python Visualize.py pretraite 1    # essais prétraités
python Visualize.py ART 1          # avant / après débruitage (2 fenêtres)
python Visualize.py ART 4 1        # avant / après, checkpoint LOSO de l'epoch 1
python CSP.py brut 1               # topographies des filtres CSP
```

---

## Les scripts

| Script | Rôle |
|--------|------|
| `ICLABEL_Brut.py` | ICA + tri ICLabel sur le signal continu 64 canaux → `Databases/EEGBCI_ICLABEL/` |
| `Pretraitement.py` | signal continu → format ART (30 canaux, 256 Hz, essais de 4 s) ; `--iclabel` pour la version nettoyée |
| `Clean_Model.py` | applique **un** modèle de débruitage → cache dans `Output/Nettoyé/` |
| `Train_Model.py` | entraîne ART (LOSO sur EEGBCI), DuoCL ou GCTNet (sur EEGdenoiseNet) |
| `Evaluation.py` | décodage gauche/droite **CSP + LDA** |
| `Metrics.py` | RMS, RMSE, MAE et SNR d'un sujet face à ICLabel |
| `ART_Epochs.py` | applique les checkpoints LOSO epoch par epoch : RMSE et accuracy |
| `mse_graph.py` | courbe train/validation d'un entraînement LOSO |
| `CSP.py` | topographies des filtres CSP, pour voir ce que le décodeur regarde |
| `Visualize.py` | inspection des signaux (brut / prétraité / débruité) |
| `Utils.py` | chargement d'un modèle, débruitage d'un essai, écriture du classeur LOSO |

> Ordre logique : `ICLABEL_Brut` → `Pretraitement` → `Train_Model` → `Clean_Model` →
> `Evaluation` / `Metrics`.
