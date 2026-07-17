"""
Entrainement des modeles DuoCL et GCTNet (debruitage EEG mono-canal) sur la
base EEGdenoiseNet, puis sauvegarde des poids dans model/<nom>/modelsave/.

--- VERSION CPU : force l'execution sur le processeur (torch CPU). ---

Ces deux modeles n'ont pas de poids pre-entraines fournis (contrairement a ART,
ICUNet, ...), il faut donc les entrainer from scratch. Le code s'inspire de
GCTNet-main/train.py (memes reseaux, meme melange signal + bruit a SNR variable).

    python Training_DuoGCT_CPU.py                      # les deux modeles, bruit Hybrid
    python Training_DuoGCT_CPU.py --model GCTNet --noise EOG
    python Training_DuoGCT_CPU.py --model DuoCL --epochs 50 --noise EMG

Base attendue : C:/Users/aymen/Desktop/ART/EEGdenoiseNet/data/
    EEG_all_epochs.npy, EOG_all_epochs.npy, EMG_all_epochs.npy   (N, 512)

Poids sauvegardes (format compatible avec Code/utils.py) :
    model/GCTNet/modelsave/checkpoint.pth.tar        (dernier epoch)
    model/GCTNet/modelsave/BEST_checkpoint.pth.tar   (meilleure val)
    model/DuoCL/modelsave/checkpoint.pth.tar
    model/DuoCL/modelsave/BEST_checkpoint.pth.tar
    (dict {"state_dict": ..., "epoch": ..., "val_rmse": ...})
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from model.GCTNet import Generator, Discriminator
from model.DuoCL import DuoCL

# ----------------------------- Chemins -----------------------------------
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
DATA_DIR = Path(r"C:\Users\aymen\Desktop\ART\EEGdenoiseNet\data")

# Ponderations des pertes adverses de GCTNet, loss_type = "feature+cls" (GCTNet-main/train.py)
W_FEATURE = 0.05      # w_f dans train.py
W_CLS = 0.05          # w_c dans train.py


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
    # test  = les derniers 10% (452 epoques), jamais vus a l'entrainement.
    # val   = 10% de la partie train, uniquement pour choisir le meilleur checkpoint.
    n = eeg.shape[0]
    n_trainfull = int(n * (1 - test_ratio))          # ~90% = 4062
    n_val = int(n_trainfull * val_ratio)             # ~406
    n_tr = n_trainfull - n_val                        # ~3656
    sl = (slice(0, n_tr), slice(n_tr, n_trainfull), slice(n_trainfull, n))
    return [(eeg[s], nos[s]) for s in sl]


class EEGwithNoise(object):
    # Genere a la volee des signaux bruites a differents SNR (-5..5 dB), cf. GCTNet-main
    def __init__(self, eeg_data, nos_data, batch_size=128):
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


# ------------------------------ Metriques ----------------------------------
def cal_snr(predict, truth):
    ps = np.sum(np.square(truth), axis=-1)
    pn = np.sum(np.square(predict - truth), axis=-1)
    return 10 * np.log10(ps / pn)


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def evaluate(model, dataset, device):
    # RMSE moyen (validation) : identique au "val_mse" de GCTNet-main/train.py qui,
    # malgre son nom, calcule bien un RMSE (.sqrt()). Sert a choisir le meilleur checkpoint.
    model.eval()
    rmses = []
    with torch.no_grad():
        for b in range(dataset.len()):
            x, y = dataset.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            p = model(x).view(x.shape[0], -1)
            rmses.append(((p - y) ** 2).mean(dim=-1).sqrt())
    return torch.cat(rmses).mean().item()


def save_ckpt(model, path, epoch, val_rmse):
    os.makedirs(path.parent, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_rmse": val_rmse}, path)


# ---------------------------- Boucles d'entrainement -----------------------
def train_gctnet(opts, train_data, val_data, save_dir, log):
    # GCTNet : generateur + discriminateur (perte MSE + feature + classification)
    device = opts.device
    model = Generator(data_num=512).to(device)
    model_d = Discriminator().to(device)
    model.apply(weights_init)
    model_d.apply(weights_init)

    opt_g = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-8)
    opt_d = torch.optim.Adam(model_d.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    best_rmse = float("inf")
    for epoch in range(opts.epochs):
        model.train()
        model_d.train()
        losses, run_sum = [], 0.0
        print(f"Epoch {epoch + 1}/{opts.epochs}  [GCTNet]")
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
            g_loss = (mse(p, y)
                      + W_FEATURE * mse(fake_f2, true_f2)
                      + W_CLS * torch.mean((fake_y - 1) ** 2))
            opt_d.zero_grad()                       # comme train.py : on remet a zero les deux
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            losses.append(g_loss.detach())
            run_sum += g_loss.item()
            pbar.set_postfix_str(f" - loss: {run_sum / (b + 1):.4f}")
        pbar.close()

        train_data.shuffle()
        train_loss = torch.stack(losses).mean().item()   # = leur train_loss (GCTNet-main)
        val_rmse = evaluate(model, val_data, device)
        improved = val_rmse < best_rmse
        tail = (f" - val_rmse improved from {best_rmse:.4f} to {val_rmse:.4f}, saving best model"
                if improved else f" - val_rmse did not improve from {best_rmse:.4f}")
        print(f"    train_loss: {train_loss:.4f} - val_rmse: {val_rmse:.4f}{tail}")
        log.write(f"[GCTNet] epoch {epoch + 1}/{opts.epochs}  "
                  f"train_loss {train_loss:.4f}  val_rmse {val_rmse:.4f}\n")
        log.flush()

        save_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_rmse)
        if improved:
            best_rmse = val_rmse
            save_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_rmse)
    return best_rmse


def train_duocl(opts, train_data, val_data, save_dir, log):
    # DuoCL : reseau de regression simple (perte MSE)
    device = opts.device
    model = DuoCL(data_num=512).to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-8)
    mse = nn.MSELoss()

    best_rmse = float("inf")
    for epoch in range(opts.epochs):
        model.train()
        losses, run_sum = [], 0.0
        print(f"Epoch {epoch + 1}/{opts.epochs}  [DuoCL]")
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
            run_sum += loss.item()
            pbar.set_postfix_str(f" - loss: {run_sum / (b + 1):.4f}")
        pbar.close()

        train_data.shuffle()
        train_loss = torch.stack(losses).mean().item()
        val_rmse = evaluate(model, val_data, device)
        improved = val_rmse < best_rmse
        tail = (f" - val_rmse improved from {best_rmse:.4f} to {val_rmse:.4f}, saving best model"
                if improved else f" - val_rmse did not improve from {best_rmse:.4f}")
        print(f"    train_loss: {train_loss:.4f} - val_rmse: {val_rmse:.4f}{tail}")
        log.write(f"[DuoCL] epoch {epoch + 1}/{opts.epochs}  "
                  f"train_loss {train_loss:.4f}  val_rmse {val_rmse:.4f}\n")
        log.flush()

        save_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_rmse)
        if improved:
            best_rmse = val_rmse
            save_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_rmse)
    return best_rmse


# --------------------------------- Test ------------------------------------
def test_report(model_name, save_dir, test_data, opts):
    # Recharge le meilleur modele et affiche RMSE relatif + correlation + SNR
    device = opts.device
    if model_name == "GCTNet":
        model = Generator(data_num=512).to(device)
    else:
        model = DuoCL(data_num=512).to(device)
    ckpt = torch.load(save_dir / "BEST_checkpoint.pth.tar", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rrmse, corr, snr = [], [], []
    with torch.no_grad():
        for b in range(test_data.len()):
            x, y = test_data.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            p = model(x).view(x.shape[0], -1)
            num = ((p - y) ** 2).mean(dim=-1).sqrt()
            den = (y ** 2).mean(dim=-1).sqrt()
            rrmse.append((num / den).cpu().numpy())
            pn, yn = p.cpu().numpy(), y.cpu().numpy()
            snr.append(cal_snr(pn, yn))
            for i in range(pn.shape[0]):
                corr.append(np.corrcoef(pn[i], yn[i])[0, 1])
    print(f"[{model_name}] TEST  rrmse {np.concatenate(rrmse).mean():.4f}  "
          f"corr {np.mean(corr):.4f}  snr {np.concatenate(snr).mean():.4f} dB")


# --------------------------------- Main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Entrainement DuoCL / GCTNet sur EEGdenoiseNet.")
    ap.add_argument("--model", choices=["DuoCL", "GCTNet", "both"], default="both")
    ap.add_argument("--noise", choices=["EOG", "EMG", "Hybrid"], default="Hybrid")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    opts = ap.parse_args()

    # --- Force l'utilisation du CPU ---
    import platform
    try:
        torch.zeros(1, device="cpu")
    except Exception as e:
        raise SystemExit(f"ERREUR : impossible d'utiliser le CPU avec cette installation "
                         f"de PyTorch ({e}). Reinstallez PyTorch.")
    opts.device = "cpu"
    print(f"Peripherique force : CPU ({platform.processor() or 'processeur'})")

    np.random.seed(0)
    torch.manual_seed(0)
    print(f"Peripherique : {opts.device} | bruit : {opts.noise} | epochs : {opts.epochs}")

    eeg, nos = load_arrays(opts.noise)
    (eeg_tr, nos_tr), (eeg_va, nos_va), (eeg_te, nos_te) = split_data(eeg, nos)
    train_data = EEGwithNoise(eeg_tr, nos_tr, opts.batch_size)
    val_data = EEGwithNoise(eeg_va, nos_va, opts.batch_size)
    test_data = EEGwithNoise(eeg_te, nos_te, opts.batch_size)
    print(f"Donnees : train {train_data.EEG_data.shape[0]}, "
          f"val {val_data.EEG_data.shape[0]}, test {test_data.EEG_data.shape[0]}")

    targets = ["DuoCL", "GCTNet"] if opts.model == "both" else [opts.model]
    for name in targets:
        save_dir = MODEL_DIR / name / "modelsave"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n===== Entrainement {name} -> {save_dir} =====")
        with open(save_dir / "model_trainValLog.txt", "a+", encoding="utf-8") as log:
            log.write(f"\n=== {name} | bruit {opts.noise} | epochs {opts.epochs} ===\n")
            if name == "GCTNet":
                best = train_gctnet(opts, train_data, val_data, save_dir, log)
            else:
                best = train_duocl(opts, train_data, val_data, save_dir, log)
            log.write(f"meilleur val_rmse : {best:.4f}\n")
        test_report(name, save_dir, test_data, opts)


if __name__ == "__main__":
    main()
