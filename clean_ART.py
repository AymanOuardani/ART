import argparse as ap
import pathlib as pl
import numpy as np
import torch
import mne as mne
from Model import tf_model, tf_data

#Chemins des fichiers
Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
Model_ART = pl.Path(r"C:\Users\aymen\Desktop\ART\Model\ART\modelsave\checkpoint.pth.tar")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Ligne de commande pour récupérer le numéro du sujet
parser = ap.ArgumentParser(description="Débruitage ART")
parser.add_argument("Sujet", type=int, help='Numéro du sujet (1-109)')
args = parser.parse_args()

#Lecture des epochs
sujet_id = "S" + str(args.Sujet).zfill(3)
fif_initial = Pretraite / (sujet_id + "-epo.fif")
epochs = mne.read_epochs(fif_initial, preload=True)

#Chargement du modèle ART
checkpoint = torch.load(Model_ART, map_location=device)
model = tf_model.make_model(30, 30, N=2).to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

#Débruitage ART essai par essai
data = epochs.get_data()                     
data_clean = np.empty_like(data)
for i in range(len(data)):
    data_noise = data[i]

    # z-score
    std = np.std(data_noise, axis=1, keepdims=True)
    avg = np.average(data_noise, axis=1, keepdims=True)
    data_noise = (data_noise - avg) / std

    # decode strategy
    with torch.no_grad():
        data_noise = torch.FloatTensor(data_noise).to(device)
        data_noise = data_noise.unsqueeze(0)
        src = data_noise
        tgt = data_noise
        batch = tf_data.Batch(src, tgt, 0)
        out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
        decode = model.generator(out)
        decode = decode.permute(0, 2, 1)
        add_tensor = torch.zeros(1, 30, 1).to(device)
        decode = torch.cat((decode, add_tensor), dim=2)
    decode = np.array(decode.cpu()).astype(np.float64)

    sortie = decode[0]
    sortie = (sortie-sortie.mean(1, keepdims=True)) / sortie.std(1, keepdims=True)
    data_clean[i] = sortie * std + avg

#Reconstruction des epochs nettoyés
epochs_clean = mne.EpochsArray(data_clean, epochs.info, epochs.events,
                               tmin=epochs.tmin, event_id=epochs.event_id)

#Sauvegarde
out = Nettoye / sujet_id / "ART_original-epo.fif"
out.parent.mkdir(parents=True, exist_ok=True)
epochs_clean.save(out, overwrite=True)
print("Signaux nettoyés (ART) sauvegardés pour Sujet ", args.Sujet)