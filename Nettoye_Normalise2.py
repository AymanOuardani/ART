import pathlib as pl
import numpy as np
import mne
mne.set_log_level("ERROR")

Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
Nettoye_Norm2 = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé_Normalisé_2")

methodes = ["DuoCL", "GCTNet", "ICUNet", "ICUNet++", "ICUNet_attn", "ART"]

for i in range(1, 110):
    sujet_id = "S" + str(i).zfill(3)
    f_ic = Nettoye / sujet_id / "ICLABEL.fif"
    if not f_ic.exists():
        continue
    iclabel = mne.read_epochs(f_ic, preload=True)
    data_ic = iclabel.get_data()

    dossier_out = Nettoye_Norm2 / sujet_id
    for m in methodes:
        f_m = Nettoye / sujet_id / (m + ".fif")
        if not f_m.exists():
            continue
        ep = mne.read_epochs(f_m, preload=True)
        if not np.array_equal(ep.events[:, 2], iclabel.events[:, 2]):
            print(sujet_id, m, ": essais non alignes, ignore")
            continue
        data_sortie = ep.get_data()
        data_new = np.empty_like(data_sortie)
        for k in range(len(data_sortie)):
            m_out, s_out = data_sortie[k].mean(), data_sortie[k].std()
            m_ic, s_ic = data_ic[k].mean(), data_ic[k].std()
            normalise = (data_sortie[k] - m_out) / s_out
            data_new[k] = s_ic * normalise + m_ic
        ep_new = mne.EpochsArray(data_new, ep.info, ep.events, tmin=ep.tmin, event_id=ep.event_id)
        dossier_out.mkdir(parents=True, exist_ok=True)
        ep_new.save(dossier_out / (m + ".fif"), overwrite=True)
    print(sujet_id, "termine")

print("TOUT_TERMINE_NORMALISE2")
