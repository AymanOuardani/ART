import pathlib as pl
import time
import numpy as np
import torch
import torch.nn as nn
import mne as mne
from torch.utils.data import TensorDataset, DataLoader
from Model import tf_model, tf_data
import Utils

mne.set_log_level("ERROR")

#Chemins des fichiers
Pretraite_Total = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité_Total")   # entrée bruitée
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")                   # cible propre (ICLABEL.fif)
Sortie = pl.Path(r"C:\Users\aymen\Desktop\ART\Model\ART_ICLABEL\modelsave")
Fichier_Excel = pl.Path(r"C:\Users\aymen\Desktop\ART\Model\ART_ICLABEL\resultats_LOSO.xlsx")
Sortie.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyperparamètres
n_epochs = 60
batch_size = 32
lr = 0.01
loss_fn = nn.MSELoss()

#1 - Charger les paires par sujet (bruité = Prétraité_Total, propre = Nettoyé/ICLABEL.fif)
sujets = {}
for s in range(1, 110):
    sujet_id = "S" + str(s).zfill(3)
    f_bruite = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
    f_propre = Nettoye / sujet_id / "ICLABEL.fif"
    if not (f_bruite.exists() and f_propre.exists()):
        continue
    X = mne.read_epochs(f_bruite, preload=True).get_data().astype(np.float32)
    Y = mne.read_epochs(f_propre, preload=True).get_data().astype(np.float32)

    #normalisation par bloc : z-score scalaire du bruité, appliqué aussi à la cible
    for i in range(len(X)):
        m, ecart = X[i].mean(), X[i].std()
        X[i] = (X[i] - m) / ecart
        Y[i] = (Y[i] - m) / ecart
    sujets[sujet_id] = (X, Y)

#2 - Leave-one-subject-out : un ART neuf entraîné sur tous les sujets sauf un, testé sur celui-ci
mses = []
for sujet_test in sujets:
    debut = time.time()
    X_tr = torch.from_numpy(np.concatenate([x for s, (x, y) in sujets.items() if s != sujet_test]))
    Y_tr = torch.from_numpy(np.concatenate([y for s, (x, y) in sujets.items() if s != sujet_test]))
    X_te, Y_te = sujets[sujet_test]
    X_te = torch.from_numpy(X_te).to(device)
    Y_te = torch.from_numpy(Y_te).to(device)

    #DataLoader : mélange + découpage en batchs, à la place de perm/idx à la main
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)

    model = tf_model.make_model(30, 30, N=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    mse_train_par_epoch, mse_eval_par_epoch, pertes_batches = [], [], []
    for ep in range(n_epochs):
        print("Sujet", sujet_test, "- Epoch", ep + 1)

        #entraînement de l'epoch
        model.train()
        pertes = []
        for i, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            batch = tf_data.Batch(src, trg, pad=0)
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            pred = model.generator(out).permute(0, 2, 1)
            perte = loss_fn(pred, trg[:, :, 1:])
            opt.zero_grad()
            perte.backward()
            opt.step()
            pertes.append(perte.item())
            print(f"    batch {i + 1}/{len(loader)} - perte {perte.item():.4f}", flush=True)
        pertes_batches.append(pertes)
        mse_train_par_epoch.append(sum(pertes) / len(pertes))

        #évaluation sur le sujet exclu, après cette epoch
        model.eval()
        with torch.no_grad():
            batch = tf_data.Batch(X_te, Y_te, pad=0)
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            pred = model.generator(out).permute(0, 2, 1)
            mse_eval = loss_fn(pred, Y_te[:, :, 1:]).item()
        mse_eval_par_epoch.append(mse_eval)

        #checkpoint de cette epoch : modelsave/SujetXXX/Epoch_NY/checkpoint.pth.tar
        dossier_epoch = Sortie / sujet_test / f"Epoch_N{ep + 1}"
        dossier_epoch.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "epoch": ep + 1, "mse_eval": mse_eval},
                   dossier_epoch / "checkpoint.pth.tar")

    Utils.sauve_feuille_loso(Fichier_Excel, sujet_test, mse_train_par_epoch, mse_eval_par_epoch, pertes_batches)

    mse = mse_eval_par_epoch[-1]
    mses.append(mse)
    print(f"  {sujet_test} : mse {mse:.4f} - {time.time() - debut:.1f}s", flush=True)

print(f"\nMSE moyenne (leave-one-subject-out) : {np.mean(mses):.4f} +/- {np.std(mses):.4f}")
