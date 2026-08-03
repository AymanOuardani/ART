"""
Entraînement des débruiteurs.

  python Train_Model.py ART        transformer ART, leave-one-subject-out sur EEGBCI
  python Train_Model.py DuoCL      DuoCL sur EEGdenoiseNet
  python Train_Model.py GCTNet     GCTNet sur EEGdenoiseNet
  python Train_Model.py all        DuoCL puis GCTNet

Options communes : --device {auto,cpu,gpu} · --gpu N · --epochs N · --batch_size N
Options EEGdenoiseNet (DuoCL / GCTNet) : --noise {EOG,EMG,Hybrid}
Options ART : --sujets S004,S007 pour ne relancer qu'une partie des folds,
              --baseline pour n'afficher que la baseline identité sans entraîner.

ART apprend à reproduire ICLabel : entrée = Output/Prétraité/SXXX_Pre.fif (signal bruité),
cible = Output/Nettoyé/SXXX/ICLABEL.fif. DuoCL et GCTNet apprennent sur EEGdenoiseNet, où
le signal propre et le bruit sont mélangés à un SNR contrôlé.

ATTENTION : relancer ART réécrit tous les checkpoints déjà présents dans Model/ART_ICLABEL.
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
Racine = pl.Path(r"C:\Users\aymen\Desktop\ART")
Model_Dir = Racine / "Model"
Pretraite = Racine / "Output" / "Prétraité"        # ART : entrée bruitée
Nettoye = Racine / "Output" / "Nettoyé"            # ART : cible propre
Cible = "ICLABEL.fif"
Sortie_ART = Model_Dir / "ART_ICLABEL" / "modelsave"
Excel_ART = Model_Dir / "ART_ICLABEL" / "resultats_LOSO.xlsx"
Data_Dir = Racine / "Databases" / "EEGdenoiseNet" / "data"
Out_Ods = Racine / "resultats_accuracy.ods"

#Hyperparamètres d'ART (reproduisent les réglages du papier)
art_n_epochs = 60
art_batch_size = 32
art_lr = 0.01
part_train = 0.8   # part des 108 autres sujets pour l'entraînement (86), le reste en validation (22)
graine = 42        # split train/validation identique d'une exécution à l'autre

#Pondérations des pertes adverses de GCTNet (loss_type "feature+cls", GCTNet-main/train.py)
W_FEATURE = 0.05
W_CLS = 0.05

#Config par modèle EEGdenoiseNet : dossier de poids + valeurs par défaut
MODELES = {
    "DuoCL":  {"dossier": "DuoCL",  "epochs": 100, "batch": 128},
    "GCTNet": {"dossier": "GCTNet", "epochs": 100, "batch": 128},
}

#Ligne de commande
parser = ap.ArgumentParser(description="Entraînement des débruiteurs (ART, DuoCL, GCTNet)",
                           formatter_class=ap.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument("Modele", choices=["ART", "DuoCL", "GCTNet", "all"],
                    help="modèle à entraîner ('all' = DuoCL + GCTNet)")
parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    dest="device_mode", help="périphérique (défaut : auto)")
parser.add_argument("--gpu", type=int, default=0, help="index du GPU CUDA (si device gpu/auto)")
parser.add_argument("--noise", choices=["EOG", "EMG", "Hybrid"], default="Hybrid",
                    help="bruit ajouté (DuoCL / GCTNet)")
parser.add_argument("--epochs", type=int, default=None, help="nb d'epochs (défaut par modèle)")
parser.add_argument("--batch_size", type=int, default=None, help="taille de batch (défaut par modèle)")
parser.add_argument("--sujets", default=None,
                    help="ART : liste de folds à traiter, ex. S004,S007 (défaut : tous)")
parser.add_argument("--baseline", action="store_true",
                    help="ART : affiche seulement les baselines identité, sans entraîner")
args = parser.parse_args()

#Sélection du périphérique (--device auto|cpu|gpu)
if args.device_mode == "cpu":
    device = torch.device("cpu")
    print("Périphérique : CPU (forcé)")
elif args.device_mode == "gpu":
    if not torch.cuda.is_available():
        raise SystemExit("ERREUR : aucun GPU CUDA disponible alors que --device gpu est demandé.")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    print(f"Périphérique : GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)}) (forcé)")
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    print(f"Périphérique : auto -> GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
else:
    device = torch.device("cpu")
    print("Périphérique : auto -> CPU (pas de GPU CUDA)")


# =============================== ART (EEGBCI, LOSO) ===============================

def reconstruit(model, src):
    # Reconstruction strictement identique à celle de l'inférence (Utils.decode_data) : le
    # décodeur reçoit le signal BRUITÉ, jamais la cible propre. Si on lui donnait la cible
    # (teacher forcing), le modèle apprendrait à la recopier — une tâche qu'il ne reverra
    # jamais au moment de nettoyer, d'où un effondrement des performances en test.
    batch = tf_data.Batch(src, src, pad=0)
    out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
    return model.generator(out).permute(0, 2, 1)


def erreur_uv(pred, trg, ecart):
    # Résidu en µV : on redonne au résidu son échelle d'origine avec l'écart-type scalaire de
    # l'essai, puis V -> µV. La moyenne n'intervient pas, elle s'annule dans la soustraction.
    # pred fait 1023 points : on le compare aux 1023 premiers points de la cible, comme à
    # l'inférence où le dernier point est complété séparément.
    return (pred - trg[:, :, :-1]) * ecart.view(-1, 1, 1) * 1e6


def rmse_identite(X, Y, S, batch_size):
    # Baseline "identité" : l'erreur qu'on obtiendrait en recopiant simplement l'entrée bruitée.
    # Même accumulation que la validation, donc directement comparable : un modèle qui ne
    # descend pas sous ce chiffre n'a rien appris d'utile.
    somme, n = 0.0, 0
    with torch.no_grad():
        for j in range(0, len(X), batch_size):
            e = erreur_uv(X[j:j + batch_size].to(device)[:, :, :-1],
                          Y[j:j + batch_size].to(device), S[j:j + batch_size].to(device))
            somme += float((e ** 2).sum())
            n += e.numel()
    return np.sqrt(somme / n)


def rmse_modele(model, X, Y, S, batch_size):
    # RMSE du modèle sur un jeu (validation ou sujet exclu), même accumulation
    model.eval()
    somme, n = 0.0, 0
    with torch.no_grad():
        for j in range(0, len(X), batch_size):
            e = erreur_uv(reconstruit(model, X[j:j + batch_size].to(device)),
                          Y[j:j + batch_size].to(device), S[j:j + batch_size].to(device))
            somme += float((e ** 2).sum())
            n += e.numel()
    return np.sqrt(somme / n)


def entraine_art():
    n_epochs = args.epochs if args.epochs is not None else art_n_epochs
    batch_size = args.batch_size if args.batch_size is not None else art_batch_size

    #1 - Charger les paires par sujet (bruité = Prétraité, propre = Nettoyé/ICLABEL.fif)
    sujets = {}
    for s in range(1, 110):
        sujet_id = "S" + str(s).zfill(3)
        f_bruite = Pretraite / (sujet_id + "_Pre.fif")
        f_propre = Nettoye / sujet_id / Cible
        if not (f_bruite.exists() and f_propre.exists()):
            continue
        X = mne.read_epochs(f_bruite, preload=True).get_data().astype(np.float32)
        Y = mne.read_epochs(f_propre, preload=True).get_data().astype(np.float32)

        #normalisation par bloc : z-score scalaire du bruité, appliqué aussi à la cible
        #(l'écart-type est gardé de côté pour redonner des µV dans la loss)
        S = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            m, ecart = X[i].mean(), X[i].std()
            X[i] = (X[i] - m) / ecart
            Y[i] = (Y[i] - m) / ecart
            S[i] = ecart
        sujets[sujet_id] = (X, Y, S)
    if not sujets:
        raise SystemExit(f"ERREUR : aucune paire trouvée dans {Pretraite} et {Nettoye}")

    #2 - Leave-one-subject-out : pour chaque sujet exclu, les 108 autres sont séparés en 86
    #sujets d'entraînement et 22 de validation (aucun sujet des deux côtés, donc la validation
    #mesure la même chose que le test : la généralisation à un sujet jamais vu). Le sujet exclu
    #ne sert qu'au test final, avec le checkpoint de la meilleure epoch de validation.
    #On commence par le sujet 4, puis tous les autres dans l'ordre.
    ordre = [s for s in ("S004",) if s in sujets] + [s for s in sujets if s != "S004"]
    if args.sujets:
        demandes = [s.strip() for s in args.sujets.split(",")]
        ordre = [s for s in ordre if s in demandes]

    Sortie_ART.mkdir(parents=True, exist_ok=True)
    rmses = []
    for sujet_test in ordre:
        debut = time.time()

        #split au niveau des sujets (et non des essais, sinon un même sujet serait des deux côtés)
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

        #Point de comparaison, avant tout entraînement : l'erreur d'un "modèle" qui recopierait
        #son entrée. Tout RMSE au-dessus de ces valeurs signale un apprentissage inutile.
        rmse_id_val = rmse_identite(X_val, Y_val, S_val, batch_size)
        rmse_id_test = rmse_identite(X_te, Y_te, S_te, batch_size)
        print(f"{sujet_test} : RMSE identité val {rmse_id_val:.2f} µV | "
              f"test {rmse_id_test:.2f} µV", flush=True)
        if args.baseline:
            continue

        loader = DataLoader(TensorDataset(X_tr, Y_tr, S_tr), batch_size=batch_size, shuffle=True)
        model = tf_model.make_model(30, 30, N=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=art_lr, betas=(0.9, 0.98), eps=1e-9)
        #Décroissance cosinus du pas : à lr constant, les poids de fin d'epoch sautent trop loin
        #d'une epoch à l'autre et la courbe de validation zigzague sans descendre. En réduisant
        #progressivement le pas, les dernières epochs affinent au lieu d'osciller.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

        rmse_train_par_epoch, rmse_val_par_epoch, pertes_batches = [], [], []
        for ep in range(n_epochs):
            print(f"Sujet {sujet_test} - Epoch {ep + 1} - lr {opt.param_groups[0]['lr']:.2e}")

            model.train()
            pertes, somme, n = [], 0.0, 0
            for i, (src, trg, ecart) in enumerate(loader):
                src, trg, ecart = src.to(device), trg.to(device), ecart.to(device)
                e = erreur_uv(reconstruit(model, src), trg, ecart)
                perte = torch.sqrt(torch.mean(e ** 2))
                opt.zero_grad()
                perte.backward()
                opt.step()
                pertes.append(perte.item())
                somme += float((e ** 2).sum())
                n += e.numel()
                print(f"    batch {i + 1}/{len(loader)} - perte {perte.item():.2f} µV", flush=True)
            pertes_batches.append(pertes)
            #RMSE global de l'epoch (somme des carrés puis racine), et non moyenne des RMSE par
            #batch, qui sous-estimerait et ne serait pas comparable au RMSE de validation
            rmse_train = np.sqrt(somme / n)
            rmse_train_par_epoch.append(rmse_train)

            rmse_val = rmse_modele(model, X_val, Y_val, S_val, batch_size)
            rmse_val_par_epoch.append(rmse_val)

            #checkpoint de cette epoch : modelsave/SujetXXX/Epoch_NY/checkpoint.pth.tar
            dossier_epoch = Sortie_ART / sujet_test / f"Epoch_N{ep + 1}"
            dossier_epoch.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": ep + 1, "rmse_val": rmse_val},
                       dossier_epoch / "checkpoint.pth.tar")

            #Excel mis à jour à chaque epoch (pas seulement à la fin du sujet) : en cas de
            #crash, on ne perd que l'epoch en cours, pas les 60 epochs déjà entraînées
            Utils.sauve_feuille_loso(Excel_ART, sujet_test, rmse_train_par_epoch,
                                     rmse_val_par_epoch, pertes_batches)

            print(f"  {sujet_test} - Epoch {ep + 1}/{n_epochs} : train {rmse_train:.2f} µV | "
                  f"validation {rmse_val:.2f} µV | identité {rmse_id_val:.2f} µV", flush=True)
            sched.step()   # après les pas de l'epoch : le pas décroît pour l'epoch suivante

        #test final sur le sujet exclu, avec le checkpoint de la meilleure epoch de validation
        #(et non le modèle de la dernière epoch, qui a pu surapprendre entre-temps)
        meilleure_epoch = int(np.argmin(rmse_val_par_epoch)) + 1
        ckpt = Sortie_ART / sujet_test / f"Epoch_N{meilleure_epoch}" / "checkpoint.pth.tar"
        model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])

        rmse_test = rmse_modele(model, X_te, Y_te, S_te, batch_size)
        rmses.append(rmse_test)
        #gain relatif sur la baseline : négatif = le modèle fait pire que recopier l'entrée
        gain = (rmse_id_test - rmse_test) / rmse_id_test * 100
        print(f"  {sujet_test} : RMSE test {rmse_test:.2f} µV (epoch {meilleure_epoch}) "
              f"- identité {rmse_id_test:.2f} µV, gain {gain:+.1f} % "
              f"- {time.time() - debut:.1f}s", flush=True)

    if rmses:
        print(f"\nRMSE moyen (leave-one-subject-out, test final, µV) : "
              f"{np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")


# ========================= DuoCL / GCTNet (EEGdenoiseNet) =========================

def tuile(arr, n, graine_bruit):
    # Mélange (graine fixe) puis réplique le bruit pour obtenir exactement n époques
    rng = np.random.RandomState(graine_bruit)
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
    # ~90% train / 10% test (4514 -> 4062/452, façon article), val = 10% du train
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


def entraine_eegdenoisenet(cibles):
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
        print(f"\n===== Entraînement {nom} -> {save_dir} (epochs {epochs}, batch {batch}) =====")
        with open(save_dir / "model_trainValLog.txt", "a+", encoding="utf-8") as log:
            log.write(f"\n=== {nom} | bruit {args.noise} | epochs {epochs} | batch {batch} ===\n")
            best = (train_gctnet if nom == "GCTNet" else train_duocl)(
                epochs, train_data, val_data, test_data, save_dir, log)
            log.write(f"meilleur val_mse : {best:.4f}\n")

        test_report_seq(nom, save_dir, test_data)


# ------------------------------------ Main -----------------------------------------
if args.Modele == "ART":
    entraine_art()
else:
    entraine_eegdenoisenet(["DuoCL", "GCTNet"] if args.Modele == "all" else [args.Modele])
