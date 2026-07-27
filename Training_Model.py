import argparse as ap
import math
import pathlib as pl

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mne as mne
from tqdm import tqdm

from Model import tf_model, tf_data
from Model.GCTNet import Generator, Discriminator
from Model.DuoCL import DuoCL

mne.set_log_level("ERROR")

#Chemins des fichiers
Model_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Model")
Data_Dir = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGdenoiseNet\data")
Pretraite_Total = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité_Total")
Nettoye_ICLABEL = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé_ICLABEL")
Out_Ods = pl.Path(r"C:\Users\aymen\Desktop\ART\resultats_accuracy.ods")

#Pondérations des pertes adverses de GCTNet (loss_type "feature+cls", GCTNet-main/train.py)
W_FEATURE = 0.05
W_CLS = 0.05

#Config par modèle EEGdenoiseNet : dossier de poids + valeurs par défaut (epochs, batch_size)
MODELES = {
    "ART":    {"dossier": "ART_EEGdenoiseNet", "epochs": 50,  "batch": 16},
    "DuoCL":  {"dossier": "DuoCL",             "epochs": 100, "batch": 128},
    "GCTNet": {"dossier": "GCTNet",            "epochs": 100, "batch": 128},
}

#Ligne de commande (on entraîne toujours sur EEGBCI, dataset EEGdenoiseNet pour ART/DuoCL/GCTNet)
parser = ap.ArgumentParser(description="Entraînement des modèles de débruitage EEGBCI")
parser.add_argument("Modele", choices=["ART", "DuoCL", "GCTNet", "ICLABEL", "all"],
                    help="modèle à entraîner ('all' = ART+DuoCL+GCTNet, ICLABEL = ART neuf sur ICLabel)")
parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    dest="device_mode", help="périphérique (défaut : auto)")
parser.add_argument("--gpu", type=int, default=0, help="index du GPU CUDA (si device gpu/auto)")
parser.add_argument("--noise", choices=["EOG", "EMG", "Hybrid"], default="Hybrid",
                    help="bruit ajouté pour ART/DuoCL/GCTNet")
parser.add_argument("--epochs", type=int, default=None, help="nb d'epochs (défaut par modèle)")
parser.add_argument("--batch_size", type=int, default=None, help="taille de batch (défaut par modèle)")
parser.add_argument("--layers", type=int, default=2, help="ART : nb de couches encodeur/décodeur (N)")
args = parser.parse_args()

#Sélection du périphérique (--device auto|cpu|gpu)
if args.device_mode == "cpu":
    device = "cpu"
    print("Périphérique : CPU (forcé)")
elif args.device_mode == "gpu":
    if not torch.cuda.is_available():
        raise SystemExit("ERREUR : aucun GPU CUDA disponible alors que --device gpu est demandé.")
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    print(f"Périphérique : GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)}) (forcé)")
else:
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
        print(f"Périphérique : auto -> GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = "cpu"
        print("Périphérique : auto -> CPU (pas de GPU CUDA)")


# --------------------- Débruitage EEGdenoiseNet (ART / DuoCL / GCTNet) ---------------------
def tuile(arr, n, graine):
    # Mélange (graine fixe) puis réplique le bruit pour obtenir exactement n époques
    rng = np.random.RandomState(graine)
    arr = arr[rng.permutation(arr.shape[0])]
    reps = int(np.ceil(n / arr.shape[0]))
    return np.tile(arr, (reps, 1))[:n]


def charge_donnees(bruit):
    # Charge les 4514 époques EEG propres + le bruit, aligné sur l'EEG et mélangé (graine fixe)
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
    # ~90% train / 10% test (4514 -> 4062/452, facon article), val = 10% du train
    n = eeg.shape[0]
    n_trainfull = int(n * (1 - test_ratio))
    n_val = int(n_trainfull * val_ratio)
    n_tr = n_trainfull - n_val
    sl = (slice(0, n_tr), slice(n_tr, n_trainfull), slice(n_trainfull, n))
    return [(eeg[s], nos[s]) for s in sl]


class EEGAvecBruit:
    # Génère à la volée des signaux bruités à différents SNR (-5..5 dB), cf. GCTNet-main
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
        return math.ceil(self.EEG.shape[0] / self.batch_size)

    def item(self, i):
        eeg, nos, snr = self.EEG[i], self.NOS[i], self.SNR[i]
        eeg_rms = np.sqrt(np.sum(eeg ** 2) / eeg.shape[0])
        nos_rms = np.sqrt(np.sum(nos ** 2) / nos.shape[0])
        coe = eeg_rms / (nos_rms * snr)
        bruite = nos * coe + eeg
        std = np.std(bruite)
        return bruite / std, eeg / std          # (bruité normalisé, propre normalisé)

    def batch(self, i):
        deb = i * self.batch_size
        fin = min((i + 1) * self.batch_size, self.EEG.shape[0])
        bruite, propre = [], []
        for k in range(deb, fin):
            b, p = self.item(k)
            bruite.append(b)
            propre.append(p)
        return np.array(bruite, dtype=np.float32), np.array(propre, dtype=np.float32)

    def melange(self):
        self.EEG = self.EEG[np.random.permutation(self.EEG.shape[0])]
        self.NOS = self.NOS[np.random.permutation(self.NOS.shape[0])]
        self.SNR = 10 ** (np.random.uniform(-5, 5, self.EEG.shape[0]) * 0.05)


def cal_snr(pred, vrai):
    ps = np.sum(np.square(vrai), axis=-1)
    pn = np.sum(np.square(pred - vrai), axis=-1)
    return 10 * np.log10(ps / pn)


def poids_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def sauve_ckpt(model, chemin, epoch, val_mse):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_mse": val_mse}, chemin)


def sauve_feuille_mse(nom_feuille, modele, base, train_mses, val_mses, test_mses):
    # Écrit/actualise une feuille de resultats_accuracy.ods (évolution du MSE par epoch)
    lignes = [["Modele", modele, "", ""], ["Base", base, "", ""],
              ["epoch", "train_mse", "val_mse", "test_mse"]]
    for i in range(len(train_mses)):
        lignes.append([i + 1, round(train_mses[i], 6), round(val_mses[i], 6), round(test_mses[i], 6)])
    nouvelle_feuille = pd.DataFrame(lignes)

    feuilles = {}
    if Out_Ods.exists():
        try:
            feuilles = pd.read_excel(Out_Ods, sheet_name=None, engine="odf", header=None)
        except Exception:
            feuilles = {}
    feuilles[nom_feuille[:31]] = nouvelle_feuille
    with pd.ExcelWriter(Out_Ods, engine="odf") as writer:
        for nom, df in feuilles.items():
            df.to_excel(writer, sheet_name=str(nom)[:31], index=False, header=False)
    print(f"  -> évolution MSE -> {Out_Ods.name} (feuille '{nom_feuille[:31]}')")


def art_forward(model, src):
    # src : (B, 1, T). Encodeur = bruité complet, décodeur = bruité décalé (masque causal)
    batch = tf_data.Batch(src, src, 0)
    out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
    return model.generator(out).permute(0, 2, 1).squeeze(1)     # (B, T-1)


def evaluate_art(model, dataset):
    # MSE moyenne (val) sur les T-1 positions prédites
    model.eval()
    mses = []
    with torch.no_grad():
        for b in range(dataset.len()):
            x, y = dataset.batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            pred = art_forward(model, x)
            mses.append(((pred - y[:, 1:]) ** 2).mean(dim=-1))
    return torch.cat(mses).mean().item()


def train_art(epochs, train_data, val_data, test_data, save_dir, log):
    model = tf_model.make_model(1, 1, N=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    mse = nn.MSELoss()

    best_mse = float("inf")
    train_mses, val_mses, test_mses = [], [], []
    for epoch in range(epochs):
        model.train()
        pertes = []
        print(f"Epoch {epoch + 1}/{epochs}  [ART]")
        pbar = tqdm(range(train_data.len()), ascii=".=",
                    bar_format="{n_fmt}/{total_fmt} [{bar:30}] - {elapsed}{postfix}")
        for b in pbar:
            x, y = train_data.batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)
            pred = art_forward(model, x)
            perte = mse(pred, y[:, 1:])
            optimizer.zero_grad()
            perte.backward()
            optimizer.step()
            pertes.append(perte.item())
            pbar.set_postfix_str(f" - mse: {sum(pertes) / len(pertes):.4f}")
        pbar.close()

        train_data.melange()
        train_mse = sum(pertes) / len(pertes)
        val_mse = evaluate_art(model, val_data)
        test_mse = evaluate_art(model, test_data)
        train_mses.append(train_mse); val_mses.append(val_mse); test_mses.append(test_mse)
        ameliore = val_mse < best_mse
        fin = (f" - val_mse improved from {best_mse:.4f} to {val_mse:.4f}, saving best model"
               if ameliore else f" - val_mse did not improve from {best_mse:.4f}")
        print(f"    train_mse: {train_mse:.4f} - val_mse: {val_mse:.4f} - test_mse: {test_mse:.4f}{fin}")
        log.write(f"[ART] epoch {epoch + 1}/{epochs}  train_mse {train_mse:.4f}  "
                  f"val_mse {val_mse:.4f}  test_mse {test_mse:.4f}\n")
        log.flush()

        sauve_ckpt(model, save_dir / "checkpoint.pth.tar", epoch, val_mse)
        if ameliore:
            best_mse = val_mse
            sauve_ckpt(model, save_dir / "BEST_checkpoint.pth.tar", epoch, val_mse)
    sauve_feuille_mse(f"ART_{args.noise}", "ART", f"EEGdenoiseNet ({args.noise})",
                      train_mses, val_mses, test_mses)
    return best_mse


def test_report_art(save_dir, test_data):
    # Recharge le meilleur ART et affiche MSE + corrélation + SNR
    model = tf_model.make_model(1, 1, N=args.layers).to(device)
    ckpt = torch.load(save_dir / "BEST_checkpoint.pth.tar", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mses, corr, snr = [], [], []
    with torch.no_grad():
        for b in range(test_data.len()):
            x, y = test_data.batch(b)
            x = torch.from_numpy(x).to(device).unsqueeze(1)
            y = torch.from_numpy(y).to(device)[:, 1:]
            pred = art_forward(model, x)
            mses.append(((pred - y) ** 2).mean(dim=-1).cpu().numpy())
            pn, yn = pred.cpu().numpy(), y.cpu().numpy()
            snr.append(cal_snr(pn, yn))
            for i in range(pn.shape[0]):
                corr.append(np.corrcoef(pn[i], yn[i])[0, 1])
    print(f"[ART] TEST  mse {np.concatenate(mses).mean():.4f}  "
          f"corr {np.mean(corr):.4f}  snr {np.concatenate(snr).mean():.4f} dB")


def evaluate_seq(model, dataset):
    # MSE moyenne (validation) pour un réseau mono-canal direct (DuoCL / GCTNet)
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
    # GCTNet : générateur + discriminateur (perte MSE + feature + classification)
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

            # --- discriminateur (identique à GCTNet-main/train.py : pas de detach) ---
            p = model(x).view(x.shape[0], -1)
            fake_y, _, _, _ = model_d(p.unsqueeze(1))
            real_y, _, _, _ = model_d(y.unsqueeze(1))
            d_loss = 0.5 * torch.mean(fake_y ** 2) + 0.5 * torch.mean((real_y - 1) ** 2)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- générateur (perte MSE + feature + classification) ---
            p = model(x).view(x.shape[0], -1)
            fake_y, _, fake_f2, _ = model_d(p.unsqueeze(1))
            _, _, true_f2, _ = model_d(y.unsqueeze(1))
            mse_p = mse(p, y)
            g_loss = (mse_p
                      + W_FEATURE * mse(fake_f2, true_f2)
                      + W_CLS * torch.mean((fake_y - 1) ** 2))
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
    sauve_feuille_mse(f"GCTNet_{args.noise}", "GCTNet", f"EEGdenoiseNet ({args.noise})",
                      train_mses, val_mses, test_mses)
    return best_mse


def train_duocl(epochs, train_data, val_data, test_data, save_dir, log):
    # DuoCL : réseau de régression simple (perte MSE)
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
    sauve_feuille_mse(f"DuoCL_{args.noise}", "DuoCL", f"EEGdenoiseNet ({args.noise})",
                      train_mses, val_mses, test_mses)
    return best_mse


def test_report_seq(nom_modele, save_dir, test_data):
    # Recharge le meilleur DuoCL/GCTNet et affiche MSE + corrélation + SNR
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


# ------------------------------------ Main -----------------------------------------
if args.Modele == "ICLABEL":
    # Entraînement d'un ART neuf sur les paires EEGBCI (bruité = Prétraité_Total, propre = Nettoyé_ICLABEL/ICA)
    n_epochs, batch_size, lr = 20, 32, 1e-4
    sortie = Model_Dir / "ART_ICLABEL" / "modelsave"
    sortie.mkdir(parents=True, exist_ok=True)

    X_list, Y_list = [], []
    for s in range(1, 110):
        sujet_id = "S" + str(s).zfill(3)
        f_bruite = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
        f_propre = Nettoye_ICLABEL / sujet_id / (sujet_id + "-ICA.fif")
        if not (f_bruite.exists() and f_propre.exists()):
            continue
        X_list.append(mne.read_epochs(f_bruite, preload=True).get_data())
        Y_list.append(mne.read_epochs(f_propre, preload=True).get_data())
    X = np.concatenate(X_list).astype(np.float32)
    Y = np.concatenate(Y_list).astype(np.float32)
    print("Paires d'entraînement :", X.shape[0])

    #Normalisation par bloc : z-score scalaire du bruité, appliqué aussi à la cible
    for i in range(len(X)):
        m, s = X[i].mean(), X[i].std()
        X[i] = (X[i] - m) / s
        Y[i] = (Y[i] - m) / s
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    model = tf_model.make_model(30, 30, N=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(X)
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            src = X[idx].to(device)                          # (B, 30, 1024) bruité
            trg = Y[idx].to(device)                          # (B, 30, 1024) propre
            batch = tf_data.Batch(src, trg, pad=0)
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            pred = model.generator(out).permute(0, 2, 1)     # (B, 30, 1023)
            cible = trg[:, :, 1:]                            # (B, 30, 1023)
            loss = loss_fn(pred, cible)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {ep + 1}/{n_epochs} - loss {total / (n // batch_size):.4f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "epoch": n_epochs}, sortie / "checkpoint.pth.tar")
    print("Modèle ART (ICLabel) entraîné sauvegardé dans", sortie)

else:
    # Entraînement EEGdenoiseNet (ART / DuoCL / GCTNet)
    cibles = ["ART", "DuoCL", "GCTNet"] if args.Modele == "all" else [args.Modele]
    print(f"Modèles : {', '.join(cibles)} | bruit : {args.noise}")

    #Données chargées / découpées une seule fois (partagées entre modèles)
    eeg, nos = charge_donnees(args.noise)
    (eeg_tr, nos_tr), (eeg_va, nos_va), (eeg_te, nos_te) = decoupe_donnees(eeg, nos)

    for nom in cibles:
        cfg = MODELES[nom]
        epochs = args.epochs if args.epochs is not None else cfg["epochs"]
        batch = args.batch_size if args.batch_size is not None else cfg["batch"]

        np.random.seed(0)          # reproductibilité par modèle (comme des lancements séparés)
        torch.manual_seed(0)
        train_data = EEGAvecBruit(eeg_tr, nos_tr, batch)
        val_data = EEGAvecBruit(eeg_va, nos_va, batch)
        test_data = EEGAvecBruit(eeg_te, nos_te, batch)

        save_dir = Model_Dir / cfg["dossier"] / "modelsave"
        save_dir.mkdir(parents=True, exist_ok=True)
        extra = f" | N {args.layers}" if nom == "ART" else ""
        print(f"\n===== Entraînement {nom} -> {save_dir} (epochs {epochs}, batch {batch}{extra}) =====")
        with open(save_dir / "model_trainValLog.txt", "a+", encoding="utf-8") as log:
            log.write(f"\n=== {nom} | bruit {args.noise} | epochs {epochs} | batch {batch}{extra} ===\n")
            if nom == "ART":
                best = train_art(epochs, train_data, val_data, test_data, save_dir, log)
            elif nom == "GCTNet":
                best = train_gctnet(epochs, train_data, val_data, test_data, save_dir, log)
            else:
                best = train_duocl(epochs, train_data, val_data, test_data, save_dir, log)
            log.write(f"meilleur val_mse : {best:.4f}\n")

        if nom == "ART":
            test_report_art(save_dir, test_data)
        else:
            test_report_seq(nom, save_dir, test_data)
