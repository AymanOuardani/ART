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

#1) Charger toutes les paires (bruité = Prétraité_Total, propre = Nettoyé/ICLABEL.fif)
X_list, Y_list = [], []
for s in range(1, 110):
    sujet_id = "S" + str(s).zfill(3)
    f_bruite = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
    f_propre = Nettoye / sujet_id / "ICLABEL.fif"
    if not (f_bruite.exists() and f_propre.exists()):
        continue
    X_list.append(mne.read_epochs(f_bruite, preload=True).get_data())   # (n, 30, 1024)
    Y_list.append(mne.read_epochs(f_propre, preload=True).get_data())
X = np.concatenate(X_list).astype(np.float32)
Y = np.concatenate(Y_list).astype(np.float32)
print("Paires d'entraînement :", X.shape[0])

#2) Normalisation par bloc : z-score scalaire du bruité, appliqué aussi à la cible
for i in range(len(X)):
    m, s = X[i].mean(), X[i].std()
    X[i] = (X[i] - m) / s
    Y[i] = (Y[i] - m) / s
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

#3) Modèle ART neuf (entraîné de zéro)
model = tf_model.make_model(30, 30, N=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
loss_fn = nn.MSELoss()

#4) Boucle d'entraînement
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

#5) Sauvegarde du checkpoint
torch.save({"state_dict": model.state_dict(), "epoch": n_epochs}, Sortie / "checkpoint.pth.tar")
print("Modèle ART (ICLabel) entraîné sauvegardé dans", Sortie)
