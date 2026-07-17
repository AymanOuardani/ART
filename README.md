# Code/ — Guide d'utilisation

Scripts du projet **ART** pour le débruitage EEG et l'évaluation *motor imagery*
(main gauche / droite) sur la base **EEGBCI (BCI2000)**, plus l'entraînement des
modèles de comparaison **DuoCL** et **GCTNet** sur **EEGdenoiseNet**.

Tous les scripts résolvent leurs chemins par rapport à leur propre emplacement
(`Path(__file__).parent`), donc les dossiers de données (`Brut/`, `Prétraité/`,
`Nettoyé/`, `Final/`) sont créés **dans `Code/`**. Tu peux les lancer depuis la
racine (`python Code/xxx.py`) ou depuis `Code/` (`python xxx.py`).

---

## 1. Installation

Python **3.10–3.13**. Depuis la racine du dépôt :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell

pip install numpy scipy pandas scikit-learn mne matplotlib tqdm einops odfpy
pip install torch                     # version CPU (par defaut)
```

> **GPU (optionnel, fortement conseillé pour l'entraînement)** : installe la
> build CUDA de PyTorch depuis <https://pytorch.org> à la place de `pip install torch`.
> Vérifie avec `python -c "import torch; print(torch.cuda.is_available())"`.

À quoi servent les dépendances :

| Paquet | Utilisé par |
|--------|-------------|
| `mne` | tous les scripts EEGBCI (téléchargement, epochs, PSD, topomap) |
| `numpy` `scipy` `pandas` | prétraitement, filtres, epochs |
| `scikit-learn` | `Evaluation.py` (CSP + LDA) |
| `matplotlib` | visualisations, `topomap_mu.py` |
| `torch` | `utils.py`, `run_clean.py`, `Training_DuoGCT_*.py` |
| `einops` | modèle **GCTNet** (`model/GCTNet.py`) |
| `tqdm` | barres de progression d'entraînement |
| `odfpy` | `Evaluation.py` écrit un fichier `.ods` |

---

## 2. Poids des modèles (`model/`)

Le dossier `model/` contient les **définitions** des réseaux (`.py`) et un
sous-dossier `model/<Nom>/modelsave/` par modèle. Les **poids** `*.pth.tar` ne
sont **pas** versionnés (`.gitignore`) :

- **ICUNet, ICUNet++, ICUNet_attn, ART** → récupère les `checkpoint.pth.tar`
  auprès de l'équipe et place-les dans
  `model/<Nom>/modelsave/checkpoint.pth.tar`. Requis par `run_clean.py`.
- **DuoCL, GCTNet** → pas de poids fournis : **on les entraîne** (section 4).
  L'entraînement écrit `model/DuoCL/modelsave/` et `model/GCTNet/modelsave/`.

Format attendu (produit par nos scripts d'entraînement, lu par `utils.py`) :
`torch.load(...)["state_dict"]`.

---

## 3. Pipeline EEGBCI (débruitage + évaluation)

Ordre d'exécution. La **1re exécution télécharge** les EDF EEGBCI via MNE
(dans `Code/Brut/`, connexion internet nécessaire).

```powershell
# (a) Explorer un sujet (infos + signaux) — optionnel
python Code/Dataset.py --subject 1 --plot

# (b) Prétraitement : 64->30 canaux, 160->256 Hz, band-pass 1-50 Hz, essais 4 s
python Code/Pretreatement.py --subjects 1-109      # -> Code/Prétraité/S###-epo.fif

# (c) Débruitage par les modèles (nécessite les poids ART/ICUNet, section 2)
python Code/run_clean.py --subjects 1-109          # -> Code/Nettoyé/S###/<model>-epo.fif

# (d) Évaluation CSP+LDA (accuracy par modèle, 10 runs holdout)
python Code/Evaluation.py                          # -> Code/resultats_accuracy.ods

# (e) Cartographie topographique bande mu (Fig 6A)
python Code/topomap_mu.py                          # -> Code/topomap_mu_comparaison.png
```

### Scripts de visualisation (inspection, à tout moment)

```powershell
python Code/visualize_brut.py       --subject 1                 # brut 64 ch / 160 Hz
python Code/visualize_pretraite.py  --subject 1                 # après prétraitement
python Code/visualize_nettoye.py    --subject 1 --model ART     # après débruitage
python Code/visualize_finale.py     --subject 1 --model ART     # reconstruit vers 64 ch -> Code/Final/
```

| Script | Rôle | Entrée → Sortie |
|--------|------|-----------------|
| `Dataset.py` | infos/plot d'un sujet EEGBCI | (téléchargement) |
| `Pretreatement.py` | prétraitement format ART | EDF → `Prétraité/` |
| `utils.py` | chargement modèle + débruitage 1 essai | (utilisé par `run_clean`) |
| `run_clean.py` | débruite tous les sujets/modèles | `Prétraité/` → `Nettoyé/` |
| `Evaluation.py` | accuracy CSP+LDA | `Prétraité/`+`Nettoyé/` → `.ods` |
| `topomap_mu.py` | topomap puissance mu | `Prétraité/`+`Nettoyé/` → `.png` |
| `visualize_*.py` | inspection des signaux | selon l'étape |

---

## 4. Entraînement DuoCL, GCTNet & ART (sur EEGdenoiseNet)

Chaque modèle a **deux versions identiques** sauf le périphérique imposé :
`*_CPU.py` force le CPU, `*_GPU.py` force le GPU CUDA (erreur claire si aucun GPU,
option `--gpu N` pour choisir la carte).

### 4.1 DuoCL & GCTNet

```powershell
# Les deux modèles, bruit Hybrid (EOG+EMG), 100 epochs
python Code/Training_DuoGCT_CPU.py

# Un seul modèle / autre bruit / moins d'epochs
python Code/Training_DuoGCT_CPU.py --model DuoCL  --noise EOG --epochs 50
python Code/Training_DuoGCT_CPU.py --model GCTNet --noise EMG

# Version GPU (nécessite PyTorch CUDA)
python Code/Training_DuoGCT_GPU.py --gpu 0
```

**Options** : `--model {DuoCL,GCTNet,both}` · `--noise {EOG,EMG,Hybrid}` ·
`--epochs N` · `--batch_size N` (· `--gpu N` pour la version GPU).

**Sorties** (dans `model/<Nom>/modelsave/`) : `checkpoint.pth.tar` (dernier epoch),
`BEST_checkpoint.pth.tar` (meilleur val), `model_trainValLog.txt` (historique).
Affichage type Keras : barre par epoch avec `loss` en direct, puis
`train_loss` / `val_rmse` et un rapport de test final (`rrmse`, `corr`, `snr`).

> ⚠️ Sur **CPU**, GCTNet est lent (~2 s/batch). Pour un test rapide :
> `python Code/Training_DuoGCT_CPU.py --model DuoCL --epochs 5`.

### 4.2 ART (réentraîné sur EEGdenoiseNet)

Les poids ART fournis (`model/ART/`) ont été entraînés sur une **autre base**.
Ces scripts réentraînent ART **from scratch** sur EEGdenoiseNet (transformer
mono-canal `make_model(1, 1, N)`, même encodeur=bruité / décodeur=bruité décalé
qu'en inférence), et sauvegardent dans un **nouveau dossier**
`model/ART_EEGdenoiseNet/modelsave/` (le dossier `model/ART/` d'origine reste
intact).

```powershell
python Code/Training_ART_CPU.py                       # bruit Hybrid, 50 epochs
python Code/Training_ART_CPU.py --noise EOG --epochs 30 --batch_size 8
python Code/Training_ART_GPU.py --gpu 0               # nécessite PyTorch CUDA
```

**Options** : `--noise {EOG,EMG,Hybrid}` · `--epochs N` · `--batch_size N` ·
`--layers N` (nombre de couches encodeur/décodeur, défaut 2) (· `--gpu N` GPU).
Même protocole 90/10 (4062/452) et même affichage type Keras que ci-dessus.

> ⚠️ ART est un **transformer lourd** (attention sur 512 pas de temps,
> ~3–4 s/batch sur CPU) : le **GPU est fortement recommandé**. Pour un test
> rapide sur CPU : `--epochs 1 --batch_size 8`.

**Protocole commun aux trois** (fidèle à l'article) : on garde les **4514**
époques EEG, le bruit (moins nombreux) est répliqué pour couvrir chaque EEG,
mélange signal+bruit à SNR variable (-5..+5 dB), découpe **90 % / 10 %** →
**4062** entraînement / **452** test. Un sous-ensemble val (10 % du train) sert à
choisir le meilleur checkpoint.

---

## 5. Ce qu'il faut adapter selon ta machine

- **Chemin EEGdenoiseNet** : en haut de `Training_DuoGCT_*.py` **et**
  `Training_ART_*.py`, la constante `DATA_DIR` pointe en dur vers
  `C:\Users\aymen\Desktop\ART\EEGdenoiseNet\data`. Modifie-la si le dépôt est
  ailleurs. Ce dossier doit contenir `EEG_all_epochs.npy`, `EOG_all_epochs.npy`,
  `EMG_all_epochs.npy` (base EEGdenoiseNet, non versionnée).
- **Poids ART/ICUNet** : à placer dans `model/<Nom>/modelsave/` (section 2)
  avant `run_clean.py`.
- **Sujets** : `SUBJECTS = range(1, 110)` dans `Evaluation.py` et `topomap_mu.py` ;
  `--subjects` dans les autres.

---

## 6. Dépannage

| Symptôme | Cause / solution |
|----------|------------------|
| `ModuleNotFoundError: einops` | `pip install einops` (requis par GCTNet) |
| `ERREUR : aucun GPU CUDA disponible` | PyTorch est en CPU → utilise `Training_DuoGCT_CPU.py`, ou installe PyTorch CUDA |
| `FileNotFoundError` sur `checkpoint.pth.tar` | poids manquants → section 2 |
| `Can't write .ods` / moteur `odf` | `pip install odfpy` |
| Téléchargement EEGBCI lent/bloqué | 1re exécution seulement ; les EDF vont dans `Code/Brut/` |
| `FileNotFoundError` sur `EEG_all_epochs.npy` | vérifie `DATA_DIR` (section 5) |

---

## Note sur `GCTNet-main/`

Dépôt de référence (Yin et al., 2023) dont s'inspire `Training_DuoGCT_*.py`
(mêmes réseaux, même mélange signal+bruit). Les données de démo `*.npy` de ce
dossier ne sont **pas** versionnées.
