"""
Entraînement des modèles.

  python Train_Model.py ART_Local
  python Train_Model.py DuoCL
  python Train_Model.py GCTNet
  python Train_Model.py all

- ART_Local apprend à reproduire ICLabel : entrée = Output/Prétraité/SXXX_Pre.fif,
cible = Output/Nettoyé/SXXX/ICLABEL.fif. Les courbes vont dans
Ressources/Train_ART_Local.xlsx, une feuille par sujet.
- DuoCL et GCTNet apprennent sur EEGdenoiseNet, leurs courbes vont dans
Ressources/Train_DuoGCT.xlsx, une feuille par modèle et par bruit.
"""

"""
English summary: trains the denoising models used in this project.
- ART_Local: leave-one-subject-out training of the ART transformer to reproduce the
  ICLabel-cleaned signal from the preprocessed (noisy) one, one model per excluded
  subject, saving a checkpoint every epoch and per-epoch RMSE curves to Excel.
- DuoCL / GCTNet: trained on the public EEGdenoiseNet dataset with synthetic EOG/EMG
  noise injected at random SNR levels, saving the best (lowest val MSE) checkpoint and
  per-epoch MSE curves to Excel.

Usage:
  python Train_Model.py <ART_Local|DuoCL|GCTNet|all>
  ("all" trains DuoCL and GCTNet.)
"""

import argparse as ap
import math
import pathlib as pl
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import mne as mne

from Model import tf_model, tf_data
from Model.GCTNet import Generator, Discriminator
from Model.DuoCL import DuoCL
import Utils

mne.set_log_level("ERROR")

#Chemins des fichiers
Racine = pl.Path(__file__).resolve().parent
Ressources = Racine / "Ressources"
Model_Dir = Racine / "Model"
Pretraite = Racine / "Output" / "Prétraité"        # ART : entrée bruitée
Nettoye = Racine / "Output" / "Nettoyé"            # ART : cible propre
Cible = "ICLABEL.fif"
Sortie_ART = Model_Dir / "ART_Local" / "modelsave"
Excel_ART = Ressources / "Train_ART_Local.xlsx"
Data_Dir = Racine / "Databases" / "EEGdenoiseNet" / "data"
Excel_DuoGCT = Ressources / "Train_DuoGCT.xlsx"

#Hyperparamètres d'ART, repris de l'entraînement d'origine
art_n_epochs = 60
art_batch_size = 32
art_lr = 0.0023
part_train = 0.8   # part des 108 autres sujets pour l'entraînement (86), le reste en validation (22)
graine = 42        # split train/validation identique d'une exécution à l'autre

#Sujets à traiter : None = tous, ou une liste
sujets_a_traiter = None

#Bruit ajouté : EOG, EMG ou Hybrid
bruit = "Hybrid"

#Pondérations des pertes adverses de GCTNet
W_feature = 0.05
W_cls = 0.05

#Dossier de poids et réglages de chaque modèle
Modeles = {
    "DuoCL":  {"dossier": "DuoCL",  "epochs": 100, "batch": 128},
    "GCTNet": {"dossier": "GCTNet", "epochs": 100, "batch": 128},
}

#Ligne de commande
parser = ap.ArgumentParser(description="Entraînement des débruiteurs (ART, DuoCL, GCTNet)")
parser.add_argument("Modele", choices=["ART_Local", "DuoCL", "GCTNet", "all"],
                    help="modèle à entraîner ('all' = DuoCL + GCTNet)")
args = parser.parse_args()

#GPU s'il y en a un, sinon CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Périphérique : GPU", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Périphérique : CPU (pas de GPU CUDA)")


#===================================== ART =====================================

def reconstruit(model, src):
    """Run the ART transformer on a noisy batch `src` (teacher-forced on itself,
    never on the target) and return the reconstructed/denoised signal."""
    #Le décodeur reçoit le signal bruité, jamais la cible : sinon il apprendrait
    #à la recopier, ce qu'il ne pourra pas faire à l'inférence.
    batch = tf_data.Batch(src, src, pad=0)
    out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
    return model.generator(out).permute(0, 2, 1)


def residu(pred, trg):
    """Return the normalized residual pred - trg (dropping the target's last point to
    match pred's length); this is what the training loss/gradient is computed on."""
    #Résidu normalisé : c'est lui qui alimente le gradient, chaque essai du même poids.
    #pred fait 1023 points, on le compare aux 1023 premiers de la cible.
    return pred - trg[:, :, :-1]


def erreur_uv(res, ecart):
    """Rescale a normalized residual back to microvolts using each trial's stored
    std `ecart`, for reporting/plotting only (not used in the loss)."""
    #Le même résidu en µV, pour les courbes seulement
    return res * ecart.view(-1, 1, 1) * 1e6


def rmse_modele(model, X, Y, S, batch_size):
    """Evaluate `model` in batches over (X, Y, S) and return (RMSE in µV, normalized
    MSE) aggregated over every sample rather than averaged per-batch."""
    #RMSE en µV et MSE normalisée sur un jeu
    model.eval()
    somme, somme_norm, n = 0.0, 0.0, 0
    with torch.no_grad():
        for j in range(0, len(X), batch_size):
            res = residu(reconstruit(model, X[j:j + batch_size].to(device)),
                         Y[j:j + batch_size].to(device))
            e = erreur_uv(res, S[j:j + batch_size].to(device))
            somme += float((e ** 2).sum())
            somme_norm += float((res ** 2).sum())
            n += e.numel()
    return np.sqrt(somme / n), somme_norm / n


def entraine_art():
    """Leave-one-subject-out training of the ART transformer: for each subject in
    turn (held out as the final test subject), train on a random 80/20 split of the
    remaining 108 subjects, checkpoint every epoch, then report test RMSE using the
    checkpoint with the lowest validation RMSE."""
    n_epochs = art_n_epochs
    batch_size = art_batch_size

    #Paires de chaque sujet : bruité dans Prétraité, propre dans Nettoyé
    sujets = {}
    for s in range(1, 110):
        sujet_id = "S" + str(s).zfill(3)
        f_bruite = Pretraite / (sujet_id + "_Pre.fif")
        f_propre = Nettoye / sujet_id / Cible
        if not (f_bruite.exists() and f_propre.exists()):
            continue
        X = mne.read_epochs(f_bruite, preload=True).get_data().astype(np.float32)
        Y = mne.read_epochs(f_propre, preload=True).get_data().astype(np.float32)

        #z-score du bruité, appliqué aussi à la cible. L'écart-type est gardé pour les µV.
        S = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            m, ecart = X[i].mean(), X[i].std()
            X[i] = (X[i] - m) / ecart
            Y[i] = (Y[i] - m) / ecart
            S[i] = ecart
        sujets[sujet_id] = (X, Y, S)
    if not sujets:
        raise SystemExit(f"ERREUR : aucune paire trouvée dans {Pretraite} et {Nettoye}")

    #Un modèle par sujet exclu : les 108 autres se partagent en 86 pour l'entraînement et 22
    #pour la validation. Le sujet exclu ne sert qu'au test final. On commence par le sujet 4.
    ordre = [s for s in ("S004",) if s in sujets] + [s for s in sujets if s != "S004"]
    if sujets_a_traiter:
        demandes = sujets_a_traiter
        ordre = [s for s in ordre if s in demandes]

    Sortie_ART.mkdir(parents=True, exist_ok=True)
    rmses = []
    for sujet_test in ordre:
        debut = time.time()

        #Séparation par sujet, jamais par essai
        autres = [s for s in sujets if s != sujet_test]
        melange = np.random.default_rng(graine).permutation(len(autres))
        n_train = int(part_train * len(autres))
        sujets_tr = [autres[i] for i in melange[:n_train]]
        sujets_val = [autres[i] for i in melange[n_train:]]

        empile = lambda noms, k: torch.from_numpy(np.concatenate([sujets[s][k] for s in noms]))
        X_tr, Y_tr, S_tr = (empile(sujets_tr, k) for k in range(3))
        X_val, Y_val, S_val = (empile(sujets_val, k) for k in range(3))
        X_te, Y_te, S_te = (torch.from_numpy(a) for a in sujets[sujet_test])

        print(f"Sujet {sujet_test} : {len(sujets_tr)} sujets train, "
              f"{len(sujets_val)} sujets validation", flush=True)

        loader = DataLoader(TensorDataset(X_tr, Y_tr, S_tr), batch_size=batch_size, shuffle=True)
        model = tf_model.make_model(30, 30, N=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=art_lr, betas=(0.9, 0.98), eps=1e-9)
        #Décroissance en 1/racine(epoch) : 2,3e-3 à la première, 3e-4 à la soixantième
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: 1 / np.sqrt(ep + 1))

        rmse_train_par_epoch, rmse_val_par_epoch, pertes_batches = [], [], []
        for ep in range(n_epochs):
            print(f"Sujet {sujet_test} - Epoch {ep + 1} - lr {opt.param_groups[0]['lr']:.2e}")

            model.train()
            pertes, somme, somme_norm, n = [], 0.0, 0.0, 0
            for i, (src, trg, ecart) in enumerate(loader):
                src, trg, ecart = src.to(device), trg.to(device), ecart.to(device)

                #MSE sur signal normalisé : la seule à produire le gradient
                res = residu(reconstruit(model, src), trg)
                perte = torch.mean(res ** 2)
                opt.zero_grad()
                perte.backward()
                opt.step()

                #Le même résidu en µV, pour les courbes
                e = erreur_uv(res.detach(), ecart)
                pertes.append(perte.item())
                somme += float((e ** 2).sum())
                somme_norm += float((res.detach() ** 2).sum())
                n += e.numel()
                print(f"    batch {i + 1}/{len(loader)} - perte {perte.item():.4f}", flush=True)
            pertes_batches.append(pertes)
            #Somme des carrés puis racine, et non moyenne des RMSE par batch
            rmse_train = np.sqrt(somme / n)
            mse_train_norm = somme_norm / n
            rmse_train_par_epoch.append(rmse_train)

            rmse_val, mse_val_norm = rmse_modele(model, X_val, Y_val, S_val, batch_size)
            rmse_val_par_epoch.append(rmse_val)

            #Checkpoint de l'epoch
            dossier_epoch = Sortie_ART / sujet_test / f"Epoch_N{ep + 1}"
            dossier_epoch.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": ep + 1, "rmse_val": rmse_val},
                       dossier_epoch / "checkpoint.pth.tar")

            #Excel écrit à chaque epoch : un plantage ne coûte que l'epoch en cours
            Utils.sauve_feuille(Excel_ART, sujet_test, rmse_train_par_epoch,
                                     rmse_val_par_epoch, pertes_batches)

            #Bilan de l'epoch : la MSE est celle qu'optimise le modèle, le RMSE la grandeur physique
            print(f"  {sujet_test} - Epoch {ep + 1}/{n_epochs} : "
                  f"MSE train {mse_train_norm:.4f} | val {mse_val_norm:.4f}   ||   "
                  f"RMSE train {rmse_train:.2f} µV | val {rmse_val:.2f} µV", flush=True)
            sched.step()   # après les pas de l'epoch : le pas décroît pour l'epoch suivante

        #Test final avec le checkpoint de la meilleure epoch de validation
        meilleure_epoch = int(np.argmin(rmse_val_par_epoch)) + 1
        ckpt = Sortie_ART / sujet_test / f"Epoch_N{meilleure_epoch}" / "checkpoint.pth.tar"
        model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])

        rmse_test, mse_test_norm = rmse_modele(model, X_te, Y_te, S_te, batch_size)
        rmses.append(rmse_test)
        print(f"  {sujet_test} : MSE test {mse_test_norm:.4f} | RMSE test {rmse_test:.2f} µV "
              f"(epoch {meilleure_epoch}) - {time.time() - debut:.1f}s", flush=True)

    if rmses:
        print(f"\nRMSE moyen (leave-one-subject-out, test final, µV) : "
              f"{np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")


#========================= DuoCL / GCTNet (EEGdenoiseNet) =========================

def tuile(arr, n, graine_bruit):
    """Shuffle `arr` (fixed seed `graine_bruit`) and tile it to exactly `n` rows, so
    a smaller noise dataset can be matched to a larger EEG dataset."""
    #Mélange (graine fixe) puis réplique le bruit pour obtenir exactement n époques
    rng = np.random.RandomState(graine_bruit)
    arr = arr[rng.permutation(arr.shape[0])]
    reps = int(np.ceil(n / arr.shape[0]))
    return np.tile(arr, (reps, 1))[:n]


def charge_donnees(bruit):
    """Load the clean EEGdenoiseNet epochs and the requested noise type ("EOG",
    "EMG", or "Hybrid" = normalized sum of both), tiled to match the EEG count, and
    return them shuffled together with a fixed seed."""
    #Charge les 4514 époques EEG propres + le bruit, aligné sur l'EEG et mélangé (graine fixe)
    eeg = np.load(Data_Dir / "EEG_all_epochs.npy").astype(np.float32)
    n = eeg.shape[0]
    if bruit == "EOG":
        nos = tuile(np.load(Data_Dir / "EOG_all_epochs.npy").astype(np.float32), n, 1)
    elif bruit == "EMG":
        nos = tuile(np.load(Data_Dir / "EMG_all_epochs.npy").astype(np.float32), n, 1)
    else:  # Hybrid : somme des deux bruits normalisés (GCTNet-main)
        emg = tuile(np.load(Data_Dir / "EMG_all_epochs.npy").astype(np.float32), n, 1)
        eog = tuile(np.load(Data_Dir / "EOG_all_epochs.npy").astype(np.float32), n, 2)
        nos = (emg / np.std(emg, axis=1, keepdims=True)
               + eog / np.std(eog, axis=1, keepdims=True)).astype(np.float32)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    return eeg[perm], nos[perm]


def decoupe_donnees(eeg, nos, test_ratio=0.1, val_ratio=0.1):
    """Split (eeg, nos) into train/val/test slices by contiguous ranges (no shuffle,
    since the arrays are already shuffled) and return a list of (eeg, nos) pairs."""
    #~90% train / 10% test (4514 -> 4062/452, façon article), val = 10% du train
    n = eeg.shape[0]
    n_trainfull = int(n * (1 - test_ratio))
    n_val = int(n_trainfull * val_ratio)
    n_tr = n_trainfull - n_val
    sl = (slice(0, n_tr), slice(n_tr, n_trainfull), slice(n_trainfull, n))
    return [(eeg[s], nos[s]) for s in sl]


class EEGAvecBruit:
    #Génère à la volée des signaux bruités à différents SNR (-5..5 dB), cf. GCTNet-main
    def __init__(self, eeg_data, nos_data, batch_size=16):
        self.EEG, self.NOS, self.SNR = [], [], []
        for val in 10 ** (0.05 * np.linspace(-5.0, 5.0, num=11)):
            self.EEG.append(eeg_data)
            self.NOS.append(nos_data)
            self.SNR.append(np.zeros(eeg_data.shape[0]) + val)
        self.EEG = np.concatenate(self.EEG, axis=0)
        self.NOS = np.concatenate(self.NOS, axis=0)
        self.SNR = np.concatenate(self.SNR, axis=0)
        self.batch_size = batch_size

    def len(self):
        """Number of batches for one pass over the (SNR-expanded) dataset."""
        return math.ceil(self.EEG.shape[0] / self.batch_size)

    def item(self, i):
        """Mix clean EEG and noise for sample `i` at its assigned SNR, and return
        (noisy, clean) both normalized by the noisy signal's std."""
        eeg, nos, snr = self.EEG[i], self.NOS[i], self.SNR[i]
        eeg_rms = np.sqrt(np.sum(eeg ** 2) / eeg.shape[0])
        nos_rms = np.sqrt(np.sum(nos ** 2) / nos.shape[0])
        coe = eeg_rms / (nos_rms * snr)
        bruite = nos * coe + eeg
        std = np.std(bruite)
        return bruite / std, eeg / std          # (bruité normalisé, propre normalisé)

    def batch(self, i):
        """Build noisy/clean batch `i` by stacking `item()` over its sample range."""
        deb = i * self.batch_size
        fin = min((i + 1) * self.batch_size, self.EEG.shape[0])
        bruite, propre = [], []
        for k in range(deb, fin):
            b, p = self.item(k)
            bruite.append(b)
            propre.append(p)
        return np.array(bruite, dtype=np.float32), np.array(propre, dtype=np.float32)

    def melange(self):
        """Reshuffle EEG/noise pairing and redraw a fresh random SNR per sample,
        called between epochs to vary the training noise mixtures."""
        self.EEG = self.EEG[np.random.permutation(self.EEG.shape[0])]
        self.NOS = self.NOS[np.random.permutation(self.NOS.shape[0])]
        self.SNR = 10 ** (np.random.uniform(-5, 5, self.EEG.shape[0]) * 0.05)


def cal_snr(pred, vrai):
    """Per-sample output SNR in dB between predicted and true signal."""
    ps = np.sum(np.square(vrai), axis=-1)
    pn = np.sum(np.square(pred - vrai), axis=-1)
    return 10 * np.log10(ps / pn)


def poids_init(m):
    """Xavier-init Conv1d weights and zero their bias; used as a model.apply() callback."""
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def sauve_ckpt(model, chemin, epoch, val_mse):
    """Save model weights plus the epoch number and validation MSE to `chemin`."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_mse": val_mse}, chemin)


def sauve_feuille_mse(nom_feuille, train_mses, val_mses, test_mses):
    """Update (or create) the DuoCL/GCTNet workbook with one sheet `nom_feuille`
    holding train/val/test MSE per epoch."""
    #Une feuille par modèle et par bruit : le MSE des trois jeux, epoch par epoch
    feuille = pd.DataFrame({"epoch": range(1, len(train_mses) + 1),
                            "train_mse": np.round(train_mses, 6),
                            "val_mse": np.round(val_mses, 6),
                            "test_mse": np.round(test_mses, 6)})

    feuilles = {}
    if Excel_DuoGCT.exists():
        try:
            feuilles = pd.read_excel(Excel_DuoGCT, sheet_name=None, engine="openpyxl")
        except Exception:
            feuilles = {}
    feuilles[nom_feuille[:31]] = feuille
    Excel_DuoGCT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(Excel_DuoGCT, engine="openpyxl") as writer:
        for nom, df in feuilles.items():
            df.to_excel(writer, sheet_name=str(nom)[:31], index=False)
    print(f"  -> évolution MSE -> {Excel_DuoGCT.name} (feuille '{nom_feuille[:31]}')")


def evaluate_seq(model, dataset):
    """Mean MSE of `model` over `dataset`, for the direct single-channel networks
    (DuoCL / GCTNet, no adversarial loss)."""
    #MSE moyenne (validation) pour un réseau mono-canal direct (DuoCL / GCTNet)
    model.eval()
    mses = []
    with torch.no_grad():
        for b in range(dataset.len()):
            x, y = dataset.batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            p = model(x).view(x.shape[0], -1)
            mses.append(((p - y) ** 2).mean(dim=-1))
    return torch.cat(mses).mean().item()


def train_gctnet(epochs, train_data, val_data, test_data, save_dir, log):
    """Train GCTNet's generator/discriminator pair (adversarial + feature + MSE
    losses), checkpointing every epoch and saving the best validation-MSE checkpoint.
    Returns the best validation MSE reached."""
    #GCTNet : générateur + discriminateur (perte MSE + feature + classification)
    model = Generator(data_num=512).to(device)
    model_d = Discriminator().to(device)
    model.apply(poids_init)
    model_d.apply(poids_init)

    opt_g = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-8)
    opt_d = torch.optim.Adam(model_d.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    best_mse = float("inf")
    train_mses, val_mses, test_mses = [], [], []
    for epoch in range(epochs):
        model.train()
        model_d.train()
        mse_pertes, run_mse = [], 0.0
        print(f"Epoch {epoch + 1}/{epochs}  [GCTNet]")
        pbar = tqdm(range(train_data.len()), ascii=".=",
                    bar_format="{n_fmt}/{total_fmt} [{bar:30}] - {elapsed}{postfix}")
        for b in pbar:
            x, y = train_data.batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)

            #--- discriminateur (identique à GCTNet-main/train.py : pas de detach) ---
            p = model(x).view(x.shape[0], -1)
            fake_y, _, _, _ = model_d(p.unsqueeze(1))
            real_y, _, _, _ = model_d(y.unsqueeze(1))
            d_loss = 0.5 * torch.mean(fake_y ** 2) + 0.5 * torch.mean((real_y - 1) ** 2)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            #--- générateur (perte MSE + feature + classification) ---
            p = model(x).view(x.shape[0], -1)
            fake_y, _, fake_f2, _ = model_d(p.unsqueeze(1))
            _, _, true_f2, _ = model_d(y.unsqueeze(1))
            mse_p = mse(p, y)
            g_loss = (mse_p
                      + W_feature * mse(fake_f2, true_f2)
                      + W_cls * torch.mean((fake_y - 1) ** 2))
            opt_d.zero_grad()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            mse_pertes.append(mse_p.detach())
            run_mse += mse_p.item()
            pbar.set_postfix_str(f" - mse: {run_mse / (b + 1):.4f}")
        pbar.close()

        train_data.melange()
        train_mse = torch.stack(mse_pertes).mean().item()
        val_mse = evaluate_seq(model, val_data)
        test_mse = evaluate_seq(model, test_data)
        train_mses.append(train_mse); val_mses.append(val_mse); test_mses.append(test_mse)
        ameliore = val_mse < best_mse
        fin = (f" - val_mse improved from {best_mse:.4f} to {val_mse:.4f}, saving best model"
               if ameliore else f" - val_mse did not improve from {best_mse:.4f}")
        print(f"    train_mse: {train_mse:.4f} - val_mse: {val_mse:.4f} - test_mse: {test_mse:.4f}{fin}")
        log.write(f"[GCTNet] epoch {epoch + 1}/{epochs}  train_mse {train_mse:.4f}  "
                  f"val_mse {val_mse:.4f}  test_mse {test_mse:.4f}\n")
        log.flush()

        sauve_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_mse)
        if ameliore:
            best_mse = val_mse
            sauve_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_mse)
    sauve_feuille_mse(f"GCTNet_{bruit}", train_mses, val_mses, test_mses)
    return best_mse


def train_duocl(epochs, train_data, val_data, test_data, save_dir, log):
    """Train DuoCL with a plain MSE regression loss, checkpointing every epoch and
    saving the best validation-MSE checkpoint. Returns the best validation MSE."""
    #DuoCL : réseau de régression simple (perte MSE)
    model = DuoCL(data_num=512).to(device)
    model.apply(poids_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-8)
    mse = nn.MSELoss()

    best_mse = float("inf")
    train_mses, val_mses, test_mses = [], [], []
    for epoch in range(epochs):
        model.train()
        pertes, run_mse = [], 0.0
        print(f"Epoch {epoch + 1}/{epochs}  [DuoCL]")
        pbar = tqdm(range(train_data.len()), ascii=".=",
                    bar_format="{n_fmt}/{total_fmt} [{bar:30}] - {elapsed}{postfix}")
        for b in pbar:
            x, y = train_data.batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            p = model(x).view(x.shape[0], -1)
            perte = mse(p, y)
            optimizer.zero_grad()
            perte.backward()
            optimizer.step()
            pertes.append(perte.detach())
            run_mse += perte.item()
            pbar.set_postfix_str(f" - mse: {run_mse / (b + 1):.4f}")
        pbar.close()

        train_data.melange()
        train_mse = torch.stack(pertes).mean().item()
        val_mse = evaluate_seq(model, val_data)
        test_mse = evaluate_seq(model, test_data)
        train_mses.append(train_mse); val_mses.append(val_mse); test_mses.append(test_mse)
        ameliore = val_mse < best_mse
        fin = (f" - val_mse improved from {best_mse:.4f} to {val_mse:.4f}, saving best model"
               if ameliore else f" - val_mse did not improve from {best_mse:.4f}")
        print(f"    train_mse: {train_mse:.4f} - val_mse: {val_mse:.4f} - test_mse: {test_mse:.4f}{fin}")
        log.write(f"[DuoCL] epoch {epoch + 1}/{epochs}  train_mse {train_mse:.4f}  "
                  f"val_mse {val_mse:.4f}  test_mse {test_mse:.4f}\n")
        log.flush()

        sauve_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_mse)
        if ameliore:
            best_mse = val_mse
            sauve_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_mse)
    sauve_feuille_mse(f"DuoCL_{bruit}", train_mses, val_mses, test_mses)
    return best_mse


def test_report_seq(nom_modele, save_dir, test_data):
    """Reload the best checkpoint for `nom_modele` (DuoCL/GCTNet) and print its
    test-set MSE, correlation, and SNR."""
    #Recharge le meilleur DuoCL/GCTNet et affiche MSE + corrélation + SNR
    model = (Generator(data_num=512) if nom_modele == "GCTNet" else DuoCL(data_num=512)).to(device)
    ckpt = torch.load(save_dir / "BEST_checkpoint.pth.tar", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mses, corr, snr = [], [], []
    with torch.no_grad():
        for b in range(test_data.len()):
            x, y = test_data.batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            p = model(x).view(x.shape[0], -1)
            mses.append(((p - y) ** 2).mean(dim=-1).cpu().numpy())
            pn, yn = p.cpu().numpy(), y.cpu().numpy()
            snr.append(cal_snr(pn, yn))
            for i in range(pn.shape[0]):
                corr.append(np.corrcoef(pn[i], yn[i])[0, 1])
    print(f"[{nom_modele}] TEST  mse {np.concatenate(mses).mean():.4f}  "
          f"corr {np.mean(corr):.4f}  snr {np.concatenate(snr).mean():.4f} dB")


def entraine_eegdenoisenet(cibles):
    """Load and split the EEGdenoiseNet data once, then train each model in `cibles`
    (DuoCL and/or GCTNet) on the shared split and report its test metrics."""
    print(f"Modèles : {', '.join(cibles)} | bruit : {bruit}")

    #Données chargées / découpées une seule fois (partagées entre modèles)
    eeg, nos = charge_donnees(bruit)
    (eeg_tr, nos_tr), (eeg_va, nos_va), (eeg_te, nos_te) = decoupe_donnees(eeg, nos)

    for nom in cibles:
        cfg = Modeles[nom]
        epochs = cfg["epochs"]
        batch = cfg["batch"]

        np.random.seed(0)          # reproductibilité par modèle (comme des lancements séparés)
        torch.manual_seed(0)
        train_data = EEGAvecBruit(eeg_tr, nos_tr, batch)
        val_data = EEGAvecBruit(eeg_va, nos_va, batch)
        test_data = EEGAvecBruit(eeg_te, nos_te, batch)

        save_dir = Model_Dir / cfg["dossier"] / "modelsave"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== Entraînement {nom} -> {save_dir} (epochs {epochs}, batch {batch}) =====")
        with open(save_dir / "model_trainValLog.txt", "a+", encoding="utf-8") as log:
            log.write(f"\n=== {nom} | bruit {bruit} | epochs {epochs} | batch {batch} ===\n")
            best = (train_gctnet if nom == "GCTNet" else train_duocl)(
                epochs, train_data, val_data, test_data, save_dir, log)
            log.write(f"meilleur val_mse : {best:.4f}\n")

        test_report_seq(nom, save_dir, test_data)


#------------------------------------ Main -----------------------------------------
if args.Modele == "ART_Local":
    entraine_art()
else:
    entraine_eegdenoisenet(["DuoCL", "GCTNet"] if args.Modele == "all" else [args.Modele])
