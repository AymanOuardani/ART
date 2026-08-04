# ART — Débruitage EEG & évaluation Motor-Imagery

Pipeline pour **débruiter** des signaux EEG d'imagerie motrice (base **EEGBCI**, imagerie
main gauche vs main droite, runs 4/8/12) avec plusieurs modèles de débruitage
(**ART, IC-U-Net, DuoCL, GCTNet, ICLabel**), puis **évaluer** l'effet du débruitage sur le
décodage gauche / droite (**CSP + LDA**). Les débruiteurs mono-canal sont entraînés sur
**EEGdenoiseNet**, ART est réentraîné sur EEGBCI, un modèle par sujet exclu.

Tous les scripts affichent leurs résultats directement dans le terminal, et prennent des
arguments positionnels uniquement.

---

## Les deux ART

Deux modèles partagent la même architecture et ne doivent pas être confondus :

| Nom | Ce qu'il est |
|---|---|
| **`ART_Orig`** | l'ART de l'article, avec les poids fournis par les auteurs, jamais réentraîné |
| **`ART_Local`** | le même réseau, réentraîné ici sur EEGBCI contre ICLabel |

`ART_Local` a un checkpoint par sujet **et par epoch**, d'où le numéro d'epoch réclamé par
les scripts qui l'utilisent.

---

## Le signal de référence : ICLabel

Dans tout le projet, **« ICLabel » désigne une seule chose** : l'ICA (infomax étendu)
décomposée sur le signal **continu à 64 canaux**, avant tout prétraitement, dont les
composantes sont triées par ICLabel — tout ce qui n'est classé ni *brain* ni *other* est
retiré. Décomposer sur le signal continu complet plutôt que sur des essais déjà découpés
et réduits à 30 canaux donne à l'ICA bien plus de données et de capteurs pour séparer les
sources, donc une identification des artefacts nettement plus fiable.

C'est ce signal qui sert **à la fois de cible d'entraînement à ART_Local et de référence**
pour toutes les comparaisons (RMSE, SNR, accuracy).

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
pip install numpy scipy pandas scikit-learn mne mne-icalabel matplotlib tqdm einops openpyxl torch
```

> `openpyxl` est requis pour les trois classeurs écrits dans `Output/` (cf. **Les classeurs**).

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
python Pretraitement.py brut 1       # -> Output/Prétraité/S001_Pre.fif
python Pretraitement.py iclabel 1    # -> Output/Nettoyé/S001/ICLABEL.fif

# 3. Débruitage -> Output/Nettoyé/S001/<modele>.fif
python Clean_Model.py ART_Orig 1
python Clean_Model.py DuoCL 5
python Clean_Model.py ART_Local 1 40   # checkpoint de l'epoch 40

# 4. Évaluation CSP + LDA (gauche vs droite)
python Evaluation.py brut 1        # un sujet
python Evaluation.py ART_Orig      # tous les sujets, moyenne finale
```

Tous les signaux comparés partent du même `Prétraité` (3 classes) ; le repos est retiré
juste avant la classification, si bien que toutes les méthodes portent sur exactement les
mêmes essais.

> Le brut et sa version ICLabel passent par le **même** script de prétraitement, donc par
> exactement le même réordonnancement de canaux, le même rééchantillonnage, le même filtre
> et le même découpage. C'est indispensable puisqu'ils sont ensuite comparés échantillon par
> échantillon (RMSE, SNR) et forment la paire d'entraînement d'ART_Local.

### Modèles disponibles

`Clean_Model.py` : `ART_Orig` · `ART_Local` · `ICUNet` · `ICUNet++` · `ICUNet_attn` ·
`DuoCL` · `GCTNet`. `Evaluation.py` accepte les mêmes, plus `brut` et `ICLABEL`.

| Modèle | Description |
|--------|--------------|
| `ART_Orig` | Transformer ART, poids d'origine |
| `ART_Local` | ART réentraîné contre ICLabel — nécessite un numéro d'epoch |
| `ICUNet` / `ICUNet++` / `ICUNet_attn` | familles IC-U-Net |
| `DuoCL` / `GCTNet` | débruiteurs mono-canal (EEGdenoiseNet) |
| `ICLABEL` | la référence (cf. plus haut) |

Les cinq premiers traitent les 30 canaux d'un coup ; `DuoCL` et `GCTNet` sont mono-canal et
s'appliquent canal par canal, sur des fenêtres de 512 points.

---

## Entraînement

```bash
python Train_Model.py ART_Local    # sur EEGBCI, cible = ICLabel
python Train_Model.py DuoCL        # sur EEGdenoiseNet
python Train_Model.py GCTNet
python Train_Model.py all          # DuoCL puis GCTNet
```

Ce qui se règle rarement est en constante en tête de `Train_Model.py` : nombre d'epochs,
taille de batch, pas d'apprentissage, sujets à traiter, type de bruit pour EEGdenoiseNet.

**ART_Local** : pour chaque sujet exclu, un modèle neuf est entraîné sur 86 des 108 autres
sujets, validé sur les 22 restants, puis testé sur le sujet exclu au checkpoint de la
meilleure epoch de validation. Aucun sujet n'est des deux côtés du partage, si bien que la
validation mesure la même chose que le test : la généralisation à un sujet jamais vu.

Réglages repris de l'entraînement d'origine d'ART : 60 epochs, batch 32, Adam, pas de départ
2,3e-3 décroissant en 1/√epoch — soit 3e-4 à la soixantième. Le gradient est alimenté par la
MSE sur signal normalisé et non par l'erreur en µV : en µV, un essai agité pesait jusqu'à
neuf fois un essai calme et dictait la descente. L'erreur en µV reste affichée, pour les
courbes seulement.

Produit un checkpoint par sujet **et par epoch** dans
`Model/ART_Local/modelsave/SXXX/Epoch_NY/`, plus les courbes dans
`Output/Train_ART_Local.xlsx`, réécrit à chaque epoch : un plantage ne coûte que l'epoch
en cours. DuoCL et GCTNet écrivent les leurs dans `Output/Train_DuoGCT.xlsx`, une feuille
par modèle et par bruit.

> **Attention** : aucune reprise. Relancer réentraîne tout depuis le début et **écrase** les
> checkpoints existants avec des poids différents.

---

## Analyse

```bash
python Metrics.py tout 1        # RMS, RMSE, MAE et SNR de chaque méthode face à ICLabel
python Metrics.py SNR 1         # une seule métrique
python MSE_Plot.py ART_Local 4  # courbe train/validation du sujet 4
python MSE_Plot.py DuoCL        # courbe train/validation sur EEGdenoiseNet
python MSE_Plot.py GCTNet
```

Le SNR se lit comme un rapport signal/bruit : 0 dB signifie que l'erreur a la même amplitude
que le signal de référence, 6 dB qu'elle en fait la moitié, 14 dB le cinquième.

`MSE_Plot.py` repère le minimum de validation en rouge. Un écart qui se creuse entre les
deux courbes signale du surapprentissage.

---

## Inspection

```bash
python Visualize.py brut 1         # signal continu, brut et nettoyé ICLabel
python Visualize.py pretraite 1    # essais prétraités
python Visualize.py ART_Orig 1     # avant / après débruitage (2 fenêtres)
python CSP.py brut 1               # topographies des filtres CSP
```

---

## Les classeurs

Trois fichiers `.xlsx` dans `Output/`, tous réécrits au fil de l'exécution.

| Fichier | Écrit par | Contenu |
|---|---|---|
| `Evaluation_Accuracies.xlsx` | `Evaluation.py` | une feuille par méthode : les 10 runs de chaque sujet, puis moyenne et écart-type |
| `Train_ART_Local.xlsx` | `Train_Model.py ART_Local` | une feuille par sujet : RMSE d'entraînement et de validation par epoch, en µV |
| `Train_DuoGCT.xlsx` | `Train_Model.py DuoCL` / `GCTNet` | une feuille par modèle et par bruit : MSE des trois jeux par epoch |

`Evaluation.py` met à jour les sujets qu'il vient de calculer et conserve les autres : on
peut donc relancer un seul sujet sans perdre la feuille.

---

## Les scripts

| Script | Rôle |
|--------|------|
| `ICLABEL_Brut.py` | ICA + tri ICLabel sur le signal continu 64 canaux → `Databases/EEGBCI_ICLABEL/` |
| `Pretraitement.py` | signal continu → format ART (30 canaux, 256 Hz, essais de 4 s) |
| `Clean_Model.py` | applique **un** modèle de débruitage → `Output/Nettoyé/` |
| `Train_Model.py` | entraîne ART_Local (sur EEGBCI), DuoCL ou GCTNet (sur EEGdenoiseNet) |
| `Evaluation.py` | décodage gauche/droite **CSP + LDA** → `Output/Evaluation_Accuracies.xlsx` |
| `Metrics.py` | RMS, RMSE, MAE et SNR d'un sujet face à ICLabel |
| `MSE_Plot.py` | courbe train/validation d'un entraînement |
| `CSP.py` | topographies des filtres CSP, pour voir ce que le décodeur regarde |
| `Visualize.py` | inspection des signaux (brut / prétraité / débruité) |
| `Utils.py` | chargement d'un modèle, débruitage d'un essai, écriture du classeur |

> Ordre logique : `ICLABEL_Brut` → `Pretraitement` → `Train_Model` → `Clean_Model` →
> `Evaluation` / `Metrics`.
