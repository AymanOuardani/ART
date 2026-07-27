import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne

mne.set_log_level("ERROR")

#Chemins des fichiers
Brut_fif = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI_fif")
Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")
Pretraite_Total = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité_Total")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
Nettoye_Total = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé_Total")

#Nom du fichier débruité selon le modèle (dans Nettoyé/SXXX/)
fichiers_apres = {"ART": "ART.fif",
                  "ICUNet": "ICUNet.fif", "ICUNet++": "ICUNet++.fif", "ICUNet_attn": "ICUNet_attn.fif",
                  "DuoCL": "DuoCL.fif", "GCTNet": "GCTNet.fif", "ICLABEL": "ICLABEL.fif"}

#Couleurs des évènements (gauche/droite/repos)
couleurs_evenements = {"gauche": "tab:blue", "droite": "tab:red", "repos": "tab:green"}

def couleurs(epochs):
    #les évènements sauvegardés sont espacés d'1 échantillon (index de bloc, pas un vrai temps) :
    #sans ça MNE les croit tous superposés et empile les étiquettes à l'affichage
    epochs.events[:, 0] = np.arange(len(epochs)) * len(epochs.times)
    return {nom: c for nom, c in couleurs_evenements.items() if nom in epochs.event_id}

#Ligne de commande : quoi visualiser + numéro du sujet
parser = ap.ArgumentParser(description="Visualisation d'un sujet")
parser.add_argument("Signal", choices=["brut", "pretraite", *fichiers_apres, "ICA"],
                    help="quoi visualiser")
parser.add_argument("Sujet", type=int, help="Numéro du sujet (1-109)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

mne.viz.set_browser_backend("matplotlib")

if args.Signal == "brut":
    #signal continu brut (une fenêtre), évènements T0/T1/T2 nommés et colorés
    raw = mne.io.read_raw_fif(Brut_fif / (sujet_id + "-raw.fif"), preload=True)
    events, eid = mne.events_from_annotations(raw)
    event_id = {"repos": eid["T0"], "gauche": eid["T1"], "droite": eid["T2"]}
    raw.plot(title="Sujet " + sujet_id + " - BRUT", events=events, event_id=event_id,
             event_color=couleurs_evenements, annotation_regex="(?!)", block=True)

elif args.Signal == "pretraite":
    #epochs prétraités (une fenêtre)
    epochs = mne.read_epochs(Pretraite / (sujet_id + "-epo.fif"), preload=True)
    epochs.plot(title="Sujet " + sujet_id + " - PRÉTRAITÉ", events=True, event_id=True,
                event_color=couleurs(epochs), block=True)

else:
    #avant (prétraité) vs après (débruité) en 2 fenêtres séparées
    if args.Signal == "ICA":
        fichier_avant = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
        fichier_apres = Nettoye_Total / sujet_id / (sujet_id + "-ICA.fif")
    elif args.Signal == "ICLABEL":
        fichier_avant = Pretraite_Total / (sujet_id + "_Pre_Total.fif")
        fichier_apres = Nettoye / sujet_id / fichiers_apres[args.Signal]
    else:
        fichier_avant = Pretraite / (sujet_id + "-epo.fif")
        fichier_apres = Nettoye / sujet_id / fichiers_apres[args.Signal]
    avant = mne.read_epochs(fichier_avant, preload=True)
    apres = mne.read_epochs(fichier_apres, preload=True)
    avant.plot(title="Sujet " + sujet_id + " - AVANT " + args.Signal, events=True, event_id=True,
               event_color=couleurs(avant), block=False)
    apres.plot(title="Sujet " + sujet_id + " - APRÈS " + args.Signal, events=True, event_id=True,
               event_color=couleurs(apres), block=True)
