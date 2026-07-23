"""
Entrainement des modeles de debruitage EEG mono-canal sur EEGdenoiseNet.
Fichier unique fusionnant les 4 anciens scripts (ART/DuoGCT x CPU/GPU) :
le modele et le peripherique sont choisis en ligne de commande.

    model   (positionnel) : ART | DuoCL | GCTNet | all
    dataset (positionnel) : EEGdenoiseNet   (les debruiteurs s'entrainent dessus)
    --device              : auto | cpu | gpu   (defaut : auto)
    --noise               : EOG | EMG | Hybrid (type de bruit ajoute ; defaut Hybrid)

Exemples :
    python Training.py ART EEGdenoiseNet
    python Training.py ART EEGdenoiseNet --device gpu --gpu 0
    python Training.py DuoCL EEGdenoiseNet --noise EOG
    python Training.py all EEGdenoiseNet --epochs 100

Protocole (commun aux 3 modeles) : 4514 epoques EEG -> 4062 train / 452 test,
melange signal + bruit a SNR variable (-5..5 dB), cf. GCTNet-main.

Base attendue : Databases/EEGdenoiseNet/data/
    EEG_all_epochs.npy, EOG_all_epochs.npy, EMG_all_epochs.npy   (N, 512)

Poids sauvegardes (format compatible avec utils.py / Clean.py) :
    model/<dossier>/modelsave/checkpoint.pth.tar        (dernier epoch)
    model/<dossier>/modelsave/BEST_checkpoint.pth.tar   (meilleure val)
    dossier : ART_EEGdenoiseNet (ART), DuoCL, GCTNet
    (dict {"state_dict": ..., "epoch": ..., "val_mse": ...})
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from Model import tf_model, tf_data
from Model.GCTNet import Generator, Discriminator
from Model.DuoCL import DuoCL

# ----------------------------- Chemins -----------------------------------
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "Model"
DATA_DIR = BASE / "Databases" / "EEGdenoiseNet" / "data"
OUT_ODS = BASE / "resultats_accuracy.ods"      # feuilles d'evolution MSE

# Ponderations des pertes adverses de GCTNet, loss_type = "feature+cls" (GCTNet-main/train.py)
W_FEATURE = 0.05
W_CLS = 0.05

# Config par modele : dossier de poids + valeurs par defaut (epochs, batch_size).
MODEL_CFG = {
    "ART":    {"folder": "ART_EEGdenoiseNet", "epochs": 50,  "batch": 16},
    "DuoCL":  {"folder": "DuoCL",             "epochs": 100, "batch": 128},
    "GCTNet": {"folder": "GCTNet",            "epochs": 100, "batch": 128},
}


# --------------------------- Donnees / bruit -------------------------------
def _tile_to(arr, n, seed):
    # Melange (seed fixe) puis replique le bruit pour obtenir exactement n epoques,
    # afin de couvrir TOUTES les epoques EEG (le bruit est reutilise s'il est moins nombreux).
    rng = np.random.RandomState(seed)
    arr = arr[rng.permutation(arr.shape[0])]
    reps = int(np.ceil(n / arr.shape[0]))
    return np.tile(arr, (reps, 1))[:n]


def load_arrays(noise_type):
    # Charge les 4514 epoques EEG propres + le bruit, aligne le bruit sur l'EEG (tiling), melange (seed fixe).
    # On garde TOUTES les epoques EEG (facon article : 4062 train + 452 test = 4514).
    eeg = np.load(DATA_DIR / "EEG_all_epochs.npy").astype(np.float32)
    n = eeg.shape[0]
    if noise_type == "EOG":
        nos = _tile_to(np.load(DATA_DIR / "EOG_all_epochs.npy").astype(np.float32), n, 1)
    elif noise_type == "EMG":
        nos = _tile_to(np.load(DATA_DIR / "EMG_all_epochs.npy").astype(np.float32), n, 1)
    elif noise_type == "Hybrid":
        emg = _tile_to(np.load(DATA_DIR / "EMG_all_epochs.npy").astype(np.float32), n, 1)
        eog = _tile_to(np.load(DATA_DIR / "EOG_all_epochs.npy").astype(np.float32), n, 2)
        # meme melange que GCTNet-main : somme des deux bruits normalises
        nos = (emg / np.std(emg, axis=1, keepdims=True)
               + eog / np.std(eog, axis=1, keepdims=True)).astype(np.float32)
    else:
        raise ValueError(f"bruit inconnu : {noise_type}")

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    return eeg[perm], nos[perm]


def split_data(eeg, nos, test_ratio=0.1, val_ratio=0.1):
    # Decoupe facon article : ~90% train / 10% test (sur les 4514 epoques -> 4062 / 452).
    # test = les derniers 10% (452 epoques), jamais vus. val = 10% du train (choix du best).
    n = eeg.shape[0]
    n_trainfull = int(n * (1 - test_ratio))          # ~90% = 4062
    n_val = int(n_trainfull * val_ratio)             # ~406
    n_tr = n_trainfull - n_val                        # ~3656
    sl = (slice(0, n_tr), slice(n_tr, n_trainfull), slice(n_trainfull, n))
    return [(eeg[s], nos[s]) for s in sl]


class EEGwithNoise(object):
    # Genere a la volee des signaux bruites a differents SNR (-5..5 dB), cf. GCTNet-main
    def __init__(self, eeg_data, nos_data, batch_size=16):
        self.EEG_data, self.NOS_data, self.SNR_value = [], [], []
        for value in 10 ** (0.05 * np.linspace(-5.0, 5.0, num=11)):
            self.EEG_data.append(eeg_data)
            self.NOS_data.append(nos_data)
            self.SNR_value.append(np.zeros(eeg_data.shape[0]) + value)
        self.EEG_data = np.concatenate(self.EEG_data, axis=0)
        self.NOS_data = np.concatenate(self.NOS_data, axis=0)
        self.SNR_value = np.concatenate(self.SNR_value, axis=0)
        self.batch_size = batch_size

    def len(self):
        return math.ceil(self.EEG_data.shape[0] / self.batch_size)

    def get_item(self, item):
        eeg = self.EEG_data[item]
        nos = self.NOS_data[item]
        snr = self.SNR_value[item]
        eeg_rms = np.sqrt(np.sum(eeg ** 2) / eeg.shape[0])
        nos_rms = np.sqrt(np.sum(nos ** 2) / nos.shape[0])
        coe = eeg_rms / (nos_rms * snr)
        noisy = nos * coe + eeg
        std = np.std(noisy)
        return noisy / std, eeg / std          # (bruite normalise, propre normalise)

    def get_batch(self, batch_id):
        start = batch_id * self.batch_size
        end = min((batch_id + 1) * self.batch_size, self.EEG_data.shape[0])
        noisy, clean = [], []
        for item in range(start, end):
            a, b = self.get_item(item)
            noisy.append(a)
            clean.append(b)
        return np.array(noisy, dtype=np.float32), np.array(clean, dtype=np.float32)

    def shuffle(self):
        self.EEG_data = self.EEG_data[np.random.permutation(self.EEG_data.shape[0])]
        self.NOS_data = self.NOS_data[np.random.permutation(self.NOS_data.shape[0])]
        self.SNR_value = 10 ** (np.random.uniform(-5, 5, self.EEG_data.shape[0]) * 0.05)


# ------------------------------ Metriques / IO -----------------------------
def cal_snr(predict, truth):
    ps = np.sum(np.square(truth), axis=-1)
    pn = np.sum(np.square(predict - truth), axis=-1)
    return 10 * np.log10(ps / pn)


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def save_ckpt(model, path, epoch, val_mse):
    os.makedirs(path.parent, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_mse": val_mse}, path)


def save_mse_sheet(sheet_name, model_name, base, train_mses, val_mses, test_mses):
    # Ecrit / actualise une feuille dans resultats_accuracy.ods (evolution du MSE par epoch),
    # en conservant les autres feuilles existantes.
    rows = [["Modele", model_name, "", ""],
            ["Base", base, "", ""],
            ["epoch", "train_mse", "val_mse", "test_mse"]]
    for i in range(len(train_mses)):
        rows.append([i + 1, round(train_mses[i], 6), round(val_mses[i], 6), round(test_mses[i], 6)])
    new_df = pd.DataFrame(rows)

    sheets = {}
    if OUT_ODS.exists():
        try:
            sheets = pd.read_excel(OUT_ODS, sheet_name=None, engine="odf", header=None)
        except Exception:
            sheets = {}
    sheets[sheet_name[:31]] = new_df
    with pd.ExcelWriter(OUT_ODS, engine="odf") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False, header=False)
    print(f"  -> evolution MSE -> {OUT_ODS.name} (feuille '{sheet_name[:31]}')")


# ============================== ART ======================================== #
def art_forward(model, src):
    # src : (B, 1, T). Encodeur = bruite complet, decodeur = bruite decale (masque causal),
    # exactement comme l'inference dans utils.py. Renvoie le debruite (B, T-1).
    batch = tf_data.Batch(src, src, 0)
    dec_in = batch.src[:, :, 1:]
    out = model.forward(batch.src, dec_in, batch.src_mask, batch.trg_mask)
    pred = model.generator(out).permute(0, 2, 1)     # (B, 1, T-1)
    return pred.squeeze(1)                            # (B, T-1)


def evaluate_art(model, dataset, device):
    # MSE moyenne (val) sur les T-1 positions predites (pas de sqrt)
    model.eval()
    mses = []
    with torch.no_grad():
        for b in range(dataset.len()):
            x, y = dataset.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)     # (B, 1, T)
            y = torch.from_numpy(y).to(device)                   # (B, T)
            pred = art_forward(model, x)                         # (B, T-1)
            mses.append(((pred - y[:, 1:]) ** 2).mean(dim=-1))
    return torch.cat(mses).mean().item()


def train_art(opts, epochs, train_data, val_data, test_data, save_dir, log):
    device = opts.device
    model = tf_model.make_model(1, 1, N=opts.layers).to(device)     # ART mono-canal (1->1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    mse = nn.MSELoss()

    best_mse = float("inf")
    train_mses, val_mses, test_mses = [], [], []
    for epoch in range(epochs):
        model.train()
        losses = []
        print(f"Epoch {epoch + 1}/{epochs}  [ART]")
        pbar = tqdm(range(train_data.len()), ascii=".=",
                    bar_format="{n_fmt}/{total_fmt} [{bar:30}] - {elapsed}{postfix}")
        for b in pbar:
            x, y = train_data.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)      # (B, 1, T)
            y = torch.from_numpy(y).to(device)                    # (B, T)
            pred = art_forward(model, x)                          # (B, T-1)
            loss = mse(pred, y[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix_str(f" - mse: {sum(losses) / len(losses):.4f}")
        pbar.close()

        train_data.shuffle()
        train_mse = sum(losses) / len(losses)
        val_mse = evaluate_art(model, val_data, device)
        test_mse = evaluate_art(model, test_data, device)
        train_mses.append(train_mse); val_mses.append(val_mse); test_mses.append(test_mse)
        improved = val_mse < best_mse
        tail = (f" - val_mse improved from {best_mse:.4f} to {val_mse:.4f}, saving best model"
                if improved else f" - val_mse did not improve from {best_mse:.4f}")
        print(f"    train_mse: {train_mse:.4f} - val_mse: {val_mse:.4f} - test_mse: {test_mse:.4f}{tail}")
        log.write(f"[ART] epoch {epoch + 1}/{epochs}  train_mse {train_mse:.4f}  "
                  f"val_mse {val_mse:.4f}  test_mse {test_mse:.4f}\n")
        log.flush()

        save_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_mse)
        if improved:
            best_mse = val_mse
            save_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_mse)
    save_mse_sheet(f"ART_{opts.noise}", "ART", f"EEGdenoiseNet ({opts.noise})",
                   train_mses, val_mses, test_mses)
    return best_mse


def test_report_art(save_dir, test_data, opts):
    # Recharge le meilleur ART et affiche MSE + correlation + SNR
    device = opts.device
    model = tf_model.make_model(1, 1, N=opts.layers).to(device)
    ckpt = torch.load(save_dir / "BEST_checkpoint.pth.tar", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mses, corr, snr = [], [], []
    with torch.no_grad():
        for b in range(test_data.len()):
            x, y = test_data.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)[:, 1:]            # (B, T-1)
            pred = art_forward(model, x)                          # (B, T-1)
            mses.append(((pred - y) ** 2).mean(dim=-1).cpu().numpy())
            pn, yn = pred.cpu().numpy(), y.cpu().numpy()
            snr.append(cal_snr(pn, yn))
            for i in range(pn.shape[0]):
                corr.append(np.corrcoef(pn[i], yn[i])[0, 1])
    print(f"[ART] TEST  mse {np.concatenate(mses).mean():.4f}  "
          f"corr {np.mean(corr):.4f}  snr {np.concatenate(snr).mean():.4f} dB")


# ========================= DuoCL / GCTNet ================================== #
def evaluate_seq(model, dataset, device):
    # MSE moyenne (validation) pour un reseau mono-canal direct (DuoCL / GCTNet).
    model.eval()
    mses = []
    with torch.no_grad():
        for b in range(dataset.len()):
            x, y = dataset.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            p = model(x).view(x.shape[0], -1)
            mses.append(((p - y) ** 2).mean(dim=-1))
    return torch.cat(mses).mean().item()


def train_gctnet(opts, epochs, train_data, val_data, test_data, save_dir, log):
    # GCTNet : generateur + discriminateur (perte MSE + feature + classification)
    device = opts.device
    model = Generator(data_num=512).to(device)
    model_d = Discriminator().to(device)
    model.apply(weights_init)
    model_d.apply(weights_init)

    opt_g = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-8)
    opt_d = torch.optim.Adam(model_d.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    best_mse = float("inf")
    train_mses, val_mses, test_mses = [], [], []
    for epoch in range(epochs):
        model.train()
        model_d.train()
        mse_losses, run_mse = [], 0.0
        print(f"Epoch {epoch + 1}/{epochs}  [GCTNet]")
        pbar = tqdm(range(train_data.len()), ascii=".=",
                    bar_format="{n_fmt}/{total_fmt} [{bar:30}] - {elapsed}{postfix}")
        for b in pbar:
            x, y = train_data.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)

            # --- discriminateur (identique a GCTNet-main/train.py : pas de detach) ---
            p = model(x).view(x.shape[0], -1)
            fake_y, _, _, _ = model_d(p.unsqueeze(1))
            real_y, _, _, _ = model_d(y.unsqueeze(1))
            d_loss = 0.5 * torch.mean(fake_y ** 2) + 0.5 * torch.mean((real_y - 1) ** 2)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- generateur (perte MSE + feature + classification) ---
            p = model(x).view(x.shape[0], -1)
            fake_y, _, fake_f2, _ = model_d(p.unsqueeze(1))
            _, _, true_f2, _ = model_d(y.unsqueeze(1))
            mse_p = mse(p, y)
            g_loss = (mse_p
                      + W_FEATURE * mse(fake_f2, true_f2)
                      + W_CLS * torch.mean((fake_y - 1) ** 2))
            opt_d.zero_grad()                       # comme train.py : on remet a zero les deux
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            mse_losses.append(mse_p.detach())        # on suit la MSE (pas le g_loss adverse)
            run_mse += mse_p.item()
            pbar.set_postfix_str(f" - mse: {run_mse / (b + 1):.4f}")
        pbar.close()

        train_data.shuffle()
        train_mse = torch.stack(mse_losses).mean().item()
        val_mse = evaluate_seq(model, val_data, device)
        test_mse = evaluate_seq(model, test_data, device)
        train_mses.append(train_mse); val_mses.append(val_mse); test_mses.append(test_mse)
        improved = val_mse < best_mse
        tail = (f" - val_mse improved from {best_mse:.4f} to {val_mse:.4f}, saving best model"
                if improved else f" - val_mse did not improve from {best_mse:.4f}")
        print(f"    train_mse: {train_mse:.4f} - val_mse: {val_mse:.4f} - test_mse: {test_mse:.4f}{tail}")
        log.write(f"[GCTNet] epoch {epoch + 1}/{epochs}  train_mse {train_mse:.4f}  "
                  f"val_mse {val_mse:.4f}  test_mse {test_mse:.4f}\n")
        log.flush()

        save_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_mse)
        if improved:
            best_mse = val_mse
            save_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_mse)
    save_mse_sheet(f"GCTNet_{opts.noise}", "GCTNet", f"EEGdenoiseNet ({opts.noise})",
                   train_mses, val_mses, test_mses)
    return best_mse


def train_duocl(opts, epochs, train_data, val_data, test_data, save_dir, log):
    # DuoCL : reseau de regression simple (perte MSE)
    device = opts.device
    model = DuoCL(data_num=512).to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-8)
    mse = nn.MSELoss()

    best_mse = float("inf")
    train_mses, val_mses, test_mses = [], [], []
    for epoch in range(epochs):
        model.train()
        losses, run_mse = [], 0.0
        print(f"Epoch {epoch + 1}/{epochs}  [DuoCL]")
        pbar = tqdm(range(train_data.len()), ascii=".=",
                    bar_format="{n_fmt}/{total_fmt} [{bar:30}] - {elapsed}{postfix}")
        for b in pbar:
            x, y = train_data.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            p = model(x).view(x.shape[0], -1)
            loss = mse(p, y)                        # perte MSE (denoise_loss_mse dans train.py)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            run_mse += loss.item()
            pbar.set_postfix_str(f" - mse: {run_mse / (b + 1):.4f}")
        pbar.close()

        train_data.shuffle()
        train_mse = torch.stack(losses).mean().item()
        val_mse = evaluate_seq(model, val_data, device)
        test_mse = evaluate_seq(model, test_data, device)
        train_mses.append(train_mse); val_mses.append(val_mse); test_mses.append(test_mse)
        improved = val_mse < best_mse
        tail = (f" - val_mse improved from {best_mse:.4f} to {val_mse:.4f}, saving best model"
                if improved else f" - val_mse did not improve from {best_mse:.4f}")
        print(f"    train_mse: {train_mse:.4f} - val_mse: {val_mse:.4f} - test_mse: {test_mse:.4f}{tail}")
        log.write(f"[DuoCL] epoch {epoch + 1}/{epochs}  train_mse {train_mse:.4f}  "
                  f"val_mse {val_mse:.4f}  test_mse {test_mse:.4f}\n")
        log.flush()

        save_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_mse)
        if improved:
            best_mse = val_mse
            save_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_mse)
    save_mse_sheet(f"DuoCL_{opts.noise}", "DuoCL", f"EEGdenoiseNet ({opts.noise})",
                   train_mses, val_mses, test_mses)
    return best_mse


def test_report_seq(model_name, save_dir, test_data, opts):
    # Recharge le meilleur DuoCL/GCTNet et affiche MSE + correlation + SNR
    device = opts.device
    model = (Generator(data_num=512) if model_name == "GCTNet" else DuoCL(data_num=512)).to(device)
    ckpt = torch.load(save_dir / "BEST_checkpoint.pth.tar", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mses, corr, snr = [], [], []
    with torch.no_grad():
        for b in range(test_data.len()):
            x, y = test_data.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            p = model(x).view(x.shape[0], -1)
            mses.append(((p - y) ** 2).mean(dim=-1).cpu().numpy())
            pn, yn = p.cpu().numpy(), y.cpu().numpy()
            snr.append(cal_snr(pn, yn))
            for i in range(pn.shape[0]):
                corr.append(np.corrcoef(pn[i], yn[i])[0, 1])
    print(f"[{model_name}] TEST  mse {np.concatenate(mses).mean():.4f}  "
          f"corr {np.mean(corr):.4f}  snr {np.concatenate(snr).mean():.4f} dB")


# --------------------------- Selection du peripherique ---------------------
def select_device(opts):
    # --device auto|cpu|gpu -> renseigne opts.device (chaine torch : "cpu" ou "cuda:N")
    if opts.device_mode == "cpu":
        try:
            torch.zeros(1, device="cpu")
        except Exception as e:
            raise SystemExit(f"ERREUR : CPU inutilisable avec cette installation PyTorch ({e}).")
        opts.device = "cpu"
        print("Peripherique : CPU (force)")
    elif opts.device_mode == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit(
                "ERREUR : aucun GPU CUDA disponible alors que --device gpu est demande.\n"
                "  - Verifiez l'installation CUDA de PyTorch, ou utilisez --device cpu / auto.")
        if opts.gpu >= torch.cuda.device_count():
            raise SystemExit(
                f"ERREUR : GPU {opts.gpu} inexistant "
                f"({torch.cuda.device_count()} GPU(s), indices 0..{torch.cuda.device_count() - 1}).")
        opts.device = f"cuda:{opts.gpu}"
        torch.cuda.set_device(opts.gpu)
        print(f"Peripherique : GPU {opts.gpu} ({torch.cuda.get_device_name(opts.gpu)}) (force)")
    else:  # auto
        if torch.cuda.is_available() and opts.gpu < torch.cuda.device_count():
            opts.device = f"cuda:{opts.gpu}"
            torch.cuda.set_device(opts.gpu)
            print(f"Peripherique : auto -> GPU {opts.gpu} ({torch.cuda.get_device_name(opts.gpu)})")
        else:
            opts.device = "cpu"
            print("Peripherique : auto -> CPU (pas de GPU CUDA)")


# --------------------------------- Main ------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Entrainement des modeles de debruitage (ART / DuoCL / GCTNet) sur EEGdenoiseNet.")
    ap.add_argument("model", choices=["ART", "DuoCL", "GCTNet", "all"],
                    help="modele a entrainer (ou 'all' pour les trois)")
    ap.add_argument("dataset", choices=["EEGdenoiseNet"],
                    help="base d'entrainement (les debruiteurs s'entrainent sur EEGdenoiseNet)")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    dest="device_mode", help="peripherique (defaut : auto)")
    ap.add_argument("--gpu", type=int, default=0, help="index du GPU CUDA (si device gpu/auto)")
    ap.add_argument("--noise", choices=["EOG", "EMG", "Hybrid"], default="Hybrid")
    ap.add_argument("--epochs", type=int, default=None,
                    help="nb d'epochs (defaut par modele : ART 50, DuoCL/GCTNet 100)")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="taille de batch (defaut par modele : ART 16, DuoCL/GCTNet 128)")
    ap.add_argument("--layers", type=int, default=2, help="ART : nb de couches encodeur/decodeur (N)")
    opts = ap.parse_args()

    select_device(opts)
    targets = ["ART", "DuoCL", "GCTNet"] if opts.model == "all" else [opts.model]
    print(f"Modeles : {', '.join(targets)} | base : {opts.dataset} | bruit : {opts.noise}")

    # Donnees chargees / decoupees une seule fois (partagees entre modeles).
    eeg, nos = load_arrays(opts.noise)
    (eeg_tr, nos_tr), (eeg_va, nos_va), (eeg_te, nos_te) = split_data(eeg, nos)

    for name in targets:
        cfg = MODEL_CFG[name]
        epochs = opts.epochs if opts.epochs is not None else cfg["epochs"]
        batch = opts.batch_size if opts.batch_size is not None else cfg["batch"]

        np.random.seed(0)          # reproductibilite par modele (comme des lancements separes)
        torch.manual_seed(0)
        train_data = EEGwithNoise(eeg_tr, nos_tr, batch)
        val_data = EEGwithNoise(eeg_va, nos_va, batch)
        test_data = EEGwithNoise(eeg_te, nos_te, batch)

        save_dir = MODEL_DIR / cfg["folder"] / "modelsave"
        os.makedirs(save_dir, exist_ok=True)
        extra = f" | N {opts.layers}" if name == "ART" else ""
        print(f"\n===== Entrainement {name} -> {save_dir} "
              f"(epochs {epochs}, batch {batch}{extra}) =====")
        with open(save_dir / "model_trainValLog.txt", "a+", encoding="utf-8") as log:
            log.write(f"\n=== {name} | bruit {opts.noise} | epochs {epochs} | batch {batch}{extra} ===\n")
            if name == "ART":
                best = train_art(opts, epochs, train_data, val_data, test_data, save_dir, log)
            elif name == "GCTNet":
                best = train_gctnet(opts, epochs, train_data, val_data, test_data, save_dir, log)
            else:
                best = train_duocl(opts, epochs, train_data, val_data, test_data, save_dir, log)
            log.write(f"meilleur val_mse : {best:.4f}\n")

        if name == "ART":
            test_report_art(save_dir, test_data, opts)
        else:
            test_report_seq(name, save_dir, test_data, opts)


if __name__ == "__main__":
    main()
