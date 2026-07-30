import pathlib as pl
import numpy as np
import mne
mne.set_log_level("ERROR")

Pretraite_Total = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité_Total")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
Nettoye_Norm = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé_Normalisé")
ART_Epochs = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\ART")
ART_Norm = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\ART_Normalisé")

methodes = ["ICLABEL", "DuoCL", "GCTNet", "ICUNet", "ICUNet++", "ICUNet_attn", "ART"]


def renormalise(data_sortie, data_brut):
    data_new = np.empty_like(data_sortie)
    for i in range(len(data_sortie)):
        m_out, s_out = data_sortie[i].mean(), data_sortie[i].std()
        m_in, s_in = data_brut[i].mean(), data_brut[i].std()
        normalise = (data_sortie[i] - m_out) / s_out
        data_new[i] = s_in * normalise + m_in
    return data_new


#--- 1) Nettoyé_Normalisé : 109 sujets x {ICLABEL, DuoCL, GCTNet, ICUNet, ICUNet++, ICUNet_attn, ART (S001 seul)} ---
for i in range(1, 110):
    sujet_id = "S" + str(i).zfill(3)
    f_brut = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
    if not f_brut.exists():
        continue
    brut = mne.read_epochs(f_brut, preload=True)
    data_brut = brut.get_data()

    dossier_out = Nettoye_Norm / sujet_id
    for m in methodes:
        f_m = Nettoye / sujet_id / (m + ".fif")
        if not f_m.exists():
            continue
        ep = mne.read_epochs(f_m, preload=True)
        if not np.array_equal(ep.events[:, 2], brut.events[:, 2]):
            print(sujet_id, m, ": essais non alignes avec le brut, ignore")
            continue
        data_new = renormalise(ep.get_data(), data_brut)
        ep_new = mne.EpochsArray(data_new, ep.info, ep.events, tmin=ep.tmin, event_id=ep.event_id)
        dossier_out.mkdir(parents=True, exist_ok=True)
        ep_new.save(dossier_out / (m + ".fif"), overwrite=True)
    print(sujet_id, "termine")

print("=== Nettoye_Normalise termine ===")

#--- 2) ART_Normalisé : S001-S005 x epochs 1-60 (checkpoints LOSO ART_ICLABEL) ---
for sujet_id in ["S001", "S002", "S003", "S004", "S005"]:
    f_brut = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
    brut = mne.read_epochs(f_brut, preload=True)
    data_brut = brut.get_data()
    dossier_out = ART_Norm / sujet_id
    dossier_out.mkdir(parents=True, exist_ok=True)
    for ep_num in range(1, 61):
        f_ep = ART_Epochs / sujet_id / f"ART_epoch{ep_num}.fif"
        if not f_ep.exists():
            continue
        epochs = mne.read_epochs(f_ep, preload=True)
        if not np.array_equal(epochs.events[:, 2], brut.events[:, 2]):
            print(sujet_id, ep_num, ": essais non alignes, ignore")
            continue
        data_new = renormalise(epochs.get_data(), data_brut)
        ep_new = mne.EpochsArray(data_new, epochs.info, epochs.events, tmin=epochs.tmin, event_id=epochs.event_id)
        ep_new.save(dossier_out / f"ART_epoch{ep_num}.fif", overwrite=True)
    print(sujet_id, "ART_ICLABEL termine")

print("TOUT_TERMINE_NORMALISE")
