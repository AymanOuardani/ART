import pathlib as pl
import numpy as np
import torch
import torch.nn as nn
import mne as mne
from Model import tf_model, tf_data

mne.set_log_level("ERROR")

#Chemins des fichiers
Pretraite_Total = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité_Total")   # entrée bruitée
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")                   # cible propre (ICLABEL.fif)
Sortie = pl.Path(r"C:\Users\aymen\Desktop\ART\Model\ART_ICLABEL\modelsave")
Sortie.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyperparamètres (ART)
n_epochs = 60
batch_size = 32
lr = 0.01
loss_fn = nn.MSELoss()

#1) Charger les paires par sujet (bruité = Prétraité_Total, propre = Nettoyé/ICLABEL.fif)
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
print("Sujets disponibles :", len(sujets))

#2) Leave-one-subject-out : un ART neuf entraîné sur tous les sujets sauf un, testé sur celui-ci
mses = []
for sujet_test in sujets:
    X_tr = torch.from_numpy(np.concatenate([x for s, (x, y) in sujets.items() if s != sujet_test]))
    Y_tr = torch.from_numpy(np.concatenate([y for s, (x, y) in sujets.items() if s != sujet_test]))
    X_te, Y_te = sujets[sujet_test]
    X_te = torch.from_numpy(X_te).to(device)
    Y_te = torch.from_numpy(Y_te).to(device)

    model = tf_model.make_model(30, 30, N=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    n = len(X_tr)
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            src = X_tr[idx].to(device)
            trg = Y_tr[idx].to(device)
            batch = tf_data.Batch(src, trg, pad=0)
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            pred = model.generator(out).permute(0, 2, 1)
            perte = loss_fn(pred, trg[:, :, 1:])
            opt.zero_grad()
            perte.backward()
            opt.step()

    #évaluation MSE sur le sujet exclu
    model.eval()
    with torch.no_grad():
        batch = tf_data.Batch(X_te, Y_te, pad=0)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        pred = model.generator(out).permute(0, 2, 1)
        mse = loss_fn(pred, Y_te[:, :, 1:]).item()
    mses.append(mse)

    #sauvegarde du checkpoint (un par sujet exclu)
    torch.save({"state_dict": model.state_dict(), "epoch": n_epochs, "mse_test": mse},
               Sortie / f"{sujet_test}.pth.tar")
    print(f"  {sujet_test} : mse {mse:.4f}", flush=True)

print(f"\nMSE moyenne (leave-one-subject-out) : {np.mean(mses):.4f} +/- {np.std(mses):.4f}")
