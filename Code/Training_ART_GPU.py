"""
Entrainement du modele ART (Artifact Removal Transformer) sur EEGdenoiseNet
(debruitage EEG mono-canal), puis sauvegarde des poids dans un NOUVEAU dossier
model/ART_EEGdenoiseNet/modelsave/.

--- VERSION GPU : force l'execution sur le GPU (CUDA). Erreur si indisponible. ---

Les poids ART deja presents (model/ART/) ont ete entraines sur une AUTRE base ;
ce script reentraine ART from scratch sur EEGdenoiseNet, avec le meme protocole
que Training_DuoGCT_* (4514 epoques -> 4062 train / 452 test, melange signal +
bruit a SNR variable). Le modele mono-canal = make_model(src=1, tgt=1, N).

    python Training_ART_GPU.py                        # bruit Hybrid, 50 epochs
    python Training_ART_GPU.py --noise EOG --epochs 30
    python Training_ART_GPU.py --noise EMG --batch_size 32 --gpu 0

Base attendue : C:/Users/aymen/Desktop/ART/EEGdenoiseNet/data/
    EEG_all_epochs.npy, EOG_all_epochs.npy, EMG_all_epochs.npy   (N, 512)

Poids sauvegardes (format compatible avec Code/utils.py) :
    model/ART_EEGdenoiseNet/modelsave/checkpoint.pth.tar        (dernier epoch)
    model/ART_EEGdenoiseNet/modelsave/BEST_checkpoint.pth.tar   (meilleure val)
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

from model import tf_model, tf_data

# ----------------------------- Chemins -----------------------------------
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
DATA_DIR = Path(r"C:\Users\aymen\Desktop\ART\EEGdenoiseNet\data")
SAVE_NAME = "ART_EEGdenoiseNet"          # nouveau dossier de poids (distinct de model/ART)


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


def save_ckpt(model, path, epoch, val_rmse):
    os.makedirs(path.parent, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_rmse": val_rmse}, path)


def art_forward(model, src):
    # src : (B, 1, T). Encodeur = bruite complet, decodeur = bruite decale (masque causal),
    # exactement comme l'inference dans utils.py. Renvoie le debruite (B, T-1).
    batch = tf_data.Batch(src, src, 0)
    dec_in = batch.src[:, :, 1:]
    out = model.forward(batch.src, dec_in, batch.src_mask, batch.trg_mask)
    pred = model.generator(out).permute(0, 2, 1)     # (B, 1, T-1)
    return pred.squeeze(1)                            # (B, T-1)


def evaluate(model, dataset, device):
    # RMSE moyen (val) sur un dataset (sur les T-1 positions predites)
    model.eval()
    rmses = []
    with torch.no_grad():
        for b in range(dataset.len()):
            x, y = dataset.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)     # (B, 1, T)
            y = torch.from_numpy(y).to(device)                   # (B, T)
            pred = art_forward(model, x)                         # (B, T-1)
            rmses.append(((pred - y[:, 1:]) ** 2).mean(dim=-1).sqrt())
    return torch.cat(rmses).mean().item()


# ---------------------------- Boucle d'entrainement ------------------------
def train_art(opts, train_data, val_data, save_dir, log):
    device = opts.device
    # ART mono-canal : src_vocab = tgt_vocab = 1 canal (EEGdenoiseNet)
    model = tf_model.make_model(1, 1, N=opts.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    mse = nn.MSELoss()

    best_rmse = float("inf")
    for epoch in range(opts.epochs):
        model.train()
        losses = []
        print(f"Epoch {epoch + 1}/{opts.epochs}  [ART]")
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
            pbar.set_postfix_str(f" - loss: {sum(losses) / len(losses):.4f}")
        pbar.close()

        train_data.shuffle()
        train_loss = sum(losses) / len(losses)
        val_rmse = evaluate(model, val_data, device)
        improved = val_rmse < best_rmse
        tail = (f" - val_rmse improved from {best_rmse:.4f} to {val_rmse:.4f}, saving best model"
                if improved else f" - val_rmse did not improve from {best_rmse:.4f}")
        print(f"    train_loss: {train_loss:.4f} - val_rmse: {val_rmse:.4f}{tail}")
        log.write(f"[ART] epoch {epoch + 1}/{opts.epochs}  "
                  f"train_loss {train_loss:.4f}  val_rmse {val_rmse:.4f}\n")
        log.flush()

        save_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_rmse)
        if improved:
            best_rmse = val_rmse
            save_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_rmse)
    return best_rmse


# --------------------------------- Test ------------------------------------
def test_report(save_dir, test_data, opts):
    # Recharge le meilleur modele et affiche RMSE relatif + correlation + SNR
    device = opts.device
    model = tf_model.make_model(1, 1, N=opts.layers).to(device)
    ckpt = torch.load(save_dir / "BEST_checkpoint.pth.tar", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rrmse, corr, snr = [], [], []
    with torch.no_grad():
        for b in range(test_data.len()):
            x, y = test_data.get_batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)[:, 1:]            # (B, T-1)
            pred = art_forward(model, x)                          # (B, T-1)
            num = ((pred - y) ** 2).mean(dim=-1).sqrt()
            den = (y ** 2).mean(dim=-1).sqrt()
            rrmse.append((num / den).cpu().numpy())
            pn, yn = pred.cpu().numpy(), y.cpu().numpy()
            snr.append(cal_snr(pn, yn))
            for i in range(pn.shape[0]):
                corr.append(np.corrcoef(pn[i], yn[i])[0, 1])
    print(f"[ART] TEST  rrmse {np.concatenate(rrmse).mean():.4f}  "
          f"corr {np.mean(corr):.4f}  snr {np.concatenate(snr).mean():.4f} dB")


# --------------------------------- Main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Entrainement ART sur EEGdenoiseNet (GPU).")
    ap.add_argument("--noise", choices=["EOG", "EMG", "Hybrid"], default="Hybrid")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2, help="nb de couches encodeur/decodeur (N)")
    ap.add_argument("--gpu", type=int, default=0, help="index du GPU CUDA a utiliser")
    opts = ap.parse_args()

    # --- Force l'utilisation du GPU (CUDA) ---
    if not torch.cuda.is_available():
        raise SystemExit(
            "ERREUR : aucun GPU CUDA disponible. Cette version force l'utilisation du GPU.\n"
            "  - Verifiez que PyTorch est installe avec le support CUDA "
            "(torch.cuda.is_available() == False actuellement).\n"
            "  - Sinon, utilisez Training_ART_CPU.py pour un entrainement sur processeur.")
    if opts.gpu >= torch.cuda.device_count():
        raise SystemExit(
            f"ERREUR : GPU {opts.gpu} inexistant "
            f"({torch.cuda.device_count()} GPU(s) detecte(s), indices 0..{torch.cuda.device_count() - 1}).")
    opts.device = f"cuda:{opts.gpu}"
    torch.cuda.set_device(opts.gpu)
    print(f"Peripherique force : GPU {opts.gpu} ({torch.cuda.get_device_name(opts.gpu)})")

    np.random.seed(0)
    torch.manual_seed(0)
    print(f"Peripherique : {opts.device} | bruit : {opts.noise} | epochs : {opts.epochs} | N : {opts.layers}")

    eeg, nos = load_arrays(opts.noise)
    (eeg_tr, nos_tr), (eeg_va, nos_va), (eeg_te, nos_te) = split_data(eeg, nos)
    train_data = EEGwithNoise(eeg_tr, nos_tr, opts.batch_size)
    val_data = EEGwithNoise(eeg_va, nos_va, opts.batch_size)
    test_data = EEGwithNoise(eeg_te, nos_te, opts.batch_size)
    print(f"Donnees : train {train_data.EEG_data.shape[0]}, "
          f"val {val_data.EEG_data.shape[0]}, test {test_data.EEG_data.shape[0]}")

    save_dir = MODEL_DIR / SAVE_NAME / "modelsave"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n===== Entrainement ART -> {save_dir} =====")
    with open(save_dir / "model_trainValLog.txt", "a+", encoding="utf-8") as log:
        log.write(f"\n=== ART | bruit {opts.noise} | epochs {opts.epochs} | N {opts.layers} ===\n")
        best = train_art(opts, train_data, val_data, save_dir, log)
        log.write(f"meilleur val_rmse : {best:.4f}\n")
    test_report(save_dir, test_data, opts)


if __name__ == "__main__":
    main()
