import pathlib as pl
import time
import numpy as np
import torch
import mne as mne
from torch.utils.data import TensorDataset, DataLoader
from Model import tf_model, tf_data
import Utils

mne.set_log_level("ERROR")

#Chemins des fichiers
#Cible = ICLABEL_Amélioré.fif (ICA ICLabel sur le signal continu à 64 canaux, cf.
#ICLABEL_Brut.py) et non ICLABEL.fif : dossiers de sortie distincts pour ne pas écraser
#l'entraînement précédent, dont les résultats sont déjà dans le rapport.
Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")   # entrée bruitée
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")                   # cible propre
Cible = "ICLABEL_Amélioré.fif"
Sortie = pl.Path(r"C:\Users\aymen\Desktop\ART\Model\ART_ICLABEL_Amélioré\modelsave")
Fichier_Excel = pl.Path(r"C:\Users\aymen\Desktop\ART\Model\ART_ICLABEL_Amélioré\resultats_LOSO.xlsx")
Sortie.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyperparamètres
n_epochs = 60
batch_size = 32
lr = 0.01
part_train = 0.8   # part des 108 autres sujets pour l'entraînement (86 sujets), le reste en validation (22)
graine = 42        # split train/validation identique d'une exécution à l'autre

#Restriction de la boucle LOSO : None = tous les sujets, ou une liste pour n'en relancer
#qu'une partie, ex. ["S004"] pour ne refaire que ce fold
sujets_a_traiter = None
#True : affiche seulement les baselines identité des sujets demandés, sans rien entraîner
baseline_seulement = False


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
    # l'essai (gardé de côté à l'étape 1), puis V -> µV. La moyenne n'intervient pas, elle
    # s'annule dans la soustraction. Même grandeur que le RMSE des tableaux du rapport.
    # pred fait 1023 points : on le compare aux 1023 premiers points de la cible, comme à
    # l'inférence où le dernier point est complété séparément.
    return (pred - trg[:, :, :-1]) * ecart.view(-1, 1, 1) * 1e6


def rmse_identite(X, Y, S, batch_size, device):
    # Baseline "identité" : l'erreur qu'on obtiendrait en recopiant simplement l'entrée bruitée
    # à la place d'une reconstruction. Même accumulation que la boucle de validation (somme des
    # carrés puis racine), donc directement comparable au RMSE du modèle : un modèle qui ne
    # descend pas sous ce chiffre n'a rien appris d'utile.
    somme, n = 0.0, 0
    with torch.no_grad():
        for j in range(0, len(X), batch_size):
            src = X[j:j + batch_size].to(device)
            trg = Y[j:j + batch_size].to(device)
            ecart = S[j:j + batch_size].to(device)
            e = erreur_uv(src[:, :, :-1], trg, ecart)
            somme += float((e ** 2).sum())
            n += e.numel()
    return np.sqrt(somme / n)


#1 - Charger les paires par sujet (bruité = Prétraité, propre = Nettoyé/<Cible>)
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

#2 - Leave-one-subject-out : pour chaque sujet exclu, les 108 autres sont séparés en 86 sujets
#d'entraînement et 22 sujets de validation (aucun sujet dans les deux, donc la validation mesure
#la même chose que le test : la généralisation à un sujet jamais vu). Le sujet exclu ne sert qu'au
#test final, avec le checkpoint de la meilleure epoch de validation.
#On commence par le sujet 4, puis tous les autres dans l'ordre
ordre = [s for s in ("S004",) if s in sujets] + [s for s in sujets if s != "S004"]
if sujets_a_traiter:
    ordre = [s for s in ordre if s in sujets_a_traiter]

rmses = []
for sujet_test in ordre:
    debut = time.time()

    #split au niveau des sujets (et non des essais, sinon un même sujet serait des deux côtés)
    autres = [s for s in sujets if s != sujet_test]
    melange = np.random.default_rng(graine).permutation(len(autres))
    n_train = int(part_train * len(autres))
    sujets_tr = [autres[i] for i in melange[:n_train]]
    sujets_val = [autres[i] for i in melange[n_train:]]

    X_tr = torch.from_numpy(np.concatenate([sujets[s][0] for s in sujets_tr]))
    Y_tr = torch.from_numpy(np.concatenate([sujets[s][1] for s in sujets_tr]))
    S_tr = torch.from_numpy(np.concatenate([sujets[s][2] for s in sujets_tr]))
    X_val = torch.from_numpy(np.concatenate([sujets[s][0] for s in sujets_val]))
    Y_val = torch.from_numpy(np.concatenate([sujets[s][1] for s in sujets_val]))
    S_val = torch.from_numpy(np.concatenate([sujets[s][2] for s in sujets_val]))

    X_te, Y_te, S_te = sujets[sujet_test]
    X_te = torch.from_numpy(X_te)   # reste sur CPU, envoyé sur GPU batch par batch (comme le train)
    Y_te = torch.from_numpy(Y_te)
    S_te = torch.from_numpy(S_te)

    print(f"Sujet {sujet_test} : {len(sujets_tr)} sujets train, {len(sujets_val)} sujets validation", flush=True)

    #Point de comparaison, avant tout entraînement : l'erreur d'un "modèle" qui recopierait
    #son entrée. Tout RMSE du modèle au-dessus de ces valeurs signale un apprentissage inutile.
    rmse_id_val = rmse_identite(X_val, Y_val, S_val, batch_size, device)
    rmse_id_test = rmse_identite(X_te, Y_te, S_te, batch_size, device)
    print(f"{sujet_test} : RMSE identité val {rmse_id_val:.2f} µV | test {rmse_id_test:.2f} µV", flush=True)
    if baseline_seulement:
        continue

    #DataLoader : mélange + découpage en batchs (bruité, propre, et l'écart-type par essai)
    loader = DataLoader(TensorDataset(X_tr, Y_tr, S_tr), batch_size=batch_size, shuffle=True)

    model = tf_model.make_model(30, 30, N=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    rmse_train_par_epoch, rmse_val_par_epoch, pertes_batches = [], [], []
    for ep in range(n_epochs):
        print("Sujet", sujet_test, "- Epoch", ep + 1)

        #entraînement de l'epoch
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
        #batch, qui sous-estimerait et ne serait pas comparable aux tableaux du rapport
        rmse_train_par_epoch.append(np.sqrt(somme / n))

        #validation sur les 22 sujets mis de côté, après cette epoch
        model.eval()
        somme, n = 0.0, 0
        with torch.no_grad():
            for j in range(0, len(X_val), batch_size):
                src = X_val[j:j + batch_size].to(device)
                trg = Y_val[j:j + batch_size].to(device)
                ecart = S_val[j:j + batch_size].to(device)
                e = erreur_uv(reconstruit(model, src), trg, ecart)
                somme += float((e ** 2).sum())
                n += e.numel()
        rmse_val = np.sqrt(somme / n)
        rmse_val_par_epoch.append(rmse_val)

        #checkpoint de cette epoch : modelsave/SujetXXX/Epoch_NY/checkpoint.pth.tar
        dossier_epoch = Sortie / sujet_test / f"Epoch_N{ep + 1}"
        dossier_epoch.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "epoch": ep + 1, "rmse_val": rmse_val},
                   dossier_epoch / "checkpoint.pth.tar")

        #Excel mis à jour à chaque epoch (pas seulement à la fin du sujet) : en cas de
        #crash, on ne perd que l'epoch en cours, pas les 60 epochs déjà entraînées
        Utils.sauve_feuille_loso(Fichier_Excel, sujet_test, rmse_train_par_epoch, rmse_val_par_epoch,
                                 pertes_batches, rmse_id_val, rmse_id_test)

    #test final sur le sujet exclu, avec le checkpoint de la meilleure epoch de validation
    #(et non le modèle de la dernière epoch, qui a pu surapprendre entre-temps)
    meilleure_epoch = int(np.argmin(rmse_val_par_epoch)) + 1
    ckpt = Sortie / sujet_test / f"Epoch_N{meilleure_epoch}" / "checkpoint.pth.tar"
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])

    model.eval()
    somme, n = 0.0, 0
    with torch.no_grad():
        for j in range(0, len(X_te), batch_size):
            src = X_te[j:j + batch_size].to(device)
            trg = Y_te[j:j + batch_size].to(device)
            ecart = S_te[j:j + batch_size].to(device)
            e = erreur_uv(reconstruit(model, src), trg, ecart)
            somme += float((e ** 2).sum())
            n += e.numel()
    rmse_test = np.sqrt(somme / n)
    rmses.append(rmse_test)
    #gain relatif sur la baseline identité : négatif = le modèle fait pire que recopier l'entrée
    gain = (rmse_id_test - rmse_test) / rmse_id_test * 100
    print(f"  {sujet_test} : RMSE test {rmse_test:.2f} µV (epoch {meilleure_epoch}) "
          f"- identité {rmse_id_test:.2f} µV, gain {gain:+.1f} % "
          f"- {time.time() - debut:.1f}s", flush=True)

if rmses:
    print(f"\nRMSE moyen (leave-one-subject-out, test final, µV) : {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")
