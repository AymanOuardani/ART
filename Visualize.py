"""
Ouvre les fenêtres MNE pour regarder un signal à l'œil, avant et après débruitage.

  python Visualize.py brut 1         signal continu, brut puis nettoyé ICLabel (2 fenêtres)
  python Visualize.py pretraite 1    essais prétraités
  python Visualize.py ART_Orig 1     avant / après débruitage (2 fenêtres)

Signaux : brut · pretraite · ART_Orig · ART_Local · ICUNet · ICUNet++ · ICUNet_attn · DuoCL · GCTNet · ICLABEL
Les essais sont colorés par classe (gauche, droite, repos).
"""

"""
English summary: opens interactive MNE browser windows to visually inspect a subject's
signal, either raw vs ICLabel-cleaned, preprocessed epochs, or before/after denoising by
a chosen model. Epochs are colored by class (left, right, rest).

Usage:
  python Visualize.py <signal> <subject>
  <signal> is one of: brut, pretraite, ART_Orig, ART_Local, ICUNet, ICUNet++,
  ICUNet_attn, DuoCL, GCTNet, ICLABEL. <subject> is 1-109.
"""

import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne

mne.set_log_level("ERROR")

#Chemins des fichiers
Racine = pl.Path(__file__).resolve().parent
Brut_fif = Racine / "Databases" / "EEGBCI"
Brut_fif_ICLABEL = Racine / "Databases" / "EEGBCI_ICLABEL"
Pretraite = Racine / "Output" / "Prétraité"
Nettoye = Racine / "Output" / "Nettoyé"

#Fichier débruité de chaque modèle
fichiers_apres = {"ART_Orig": "ART_Orig.fif", "ART_Local": "ART_Local.fif",
                  "ICUNet": "ICUNet.fif", "ICUNet++": "ICUNet++.fif", "ICUNet_attn": "ICUNet_attn.fif",
                  "DuoCL": "DuoCL.fif", "GCTNet": "GCTNet.fif",
                  "ICLABEL": "ICLABEL.fif"}   # ICLabel = ICA sur le raw 64 canaux

#Couleurs des évènements
couleurs_evenements = {"gauche": "tab:blue", "droite": "tab:red", "repos": "tab:green"}

#Ligne de commande
signaux = ["brut", "pretraite", *fichiers_apres]
parser = ap.ArgumentParser(description="Visualisation d'un sujet")
parser.add_argument("Signal", help="quoi visualiser (insensible à la casse) : " + ", ".join(signaux))
parser.add_argument("Sujet", type=int, help="Numéro du sujet (1-109)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

#Insensible à la casse
correspondance = {nom.lower(): nom for nom in signaux}
if args.Signal.lower() not in correspondance:
    raise SystemExit(f"ERREUR : signal inconnu '{args.Signal}'. Choix possibles : {', '.join(signaux)}")
args.Signal = correspondance[args.Signal.lower()]

mne.viz.set_browser_backend("matplotlib")

if args.Signal == "brut":
    #Signal continu brut, puis sa version ICLabel si elle existe
    raw = mne.io.read_raw_fif(Brut_fif / (sujet_id + "-raw.fif"), preload=True)
    events, eid = mne.events_from_annotations(raw)
    event_id = {"repos": eid["T0"], "gauche": eid["T1"], "droite": eid["T2"]}

    fichier_iclabel = Brut_fif_ICLABEL / (sujet_id + "-raw.fif")
    if not fichier_iclabel.exists():
        print("Pas de version ICLabel pour", sujet_id, ":", fichier_iclabel)
    raw.plot(title="Sujet " + sujet_id + " - BRUT", events=events, event_id=event_id,
             event_color=couleurs_evenements, annotation_regex="(?!)",
             block=not fichier_iclabel.exists())

    if fichier_iclabel.exists():
        raw_iclabel = mne.io.read_raw_fif(fichier_iclabel, preload=True)
        events, eid = mne.events_from_annotations(raw_iclabel)
        event_id = {"repos": eid["T0"], "gauche": eid["T1"], "droite": eid["T2"]}
        raw_iclabel.plot(title="Sujet " + sujet_id + " - BRUT NETTOYÉ ICLABEL",
                         events=events, event_id=event_id,
                         event_color=couleurs_evenements, annotation_regex="(?!)", block=True)

elif args.Signal == "pretraite":
    #Essais prétraités
    epochs = mne.read_epochs(Pretraite / (sujet_id + "_Pre.fif"), preload=True)

    #Sans cet espacement, MNE croit les essais superposés et empile les étiquettes
    epochs.events[:, 0] = np.arange(len(epochs)) * len(epochs.times)
    couleurs = {nom: c for nom, c in couleurs_evenements.items() if nom in epochs.event_id}

    epochs.plot(title="Sujet " + sujet_id + " - PRÉTRAITÉ", events=True, event_id=True,
                event_color=couleurs, block=True)

else:
    #Avant et après débruitage, en deux fenêtres
    fichier_avant = Pretraite / (sujet_id + "_Pre.fif")
    fichier_apres = Nettoye / sujet_id / fichiers_apres[args.Signal]
    titre_apres = "APRÈS " + args.Signal
    avant = mne.read_epochs(fichier_avant, preload=True)
    apres = mne.read_epochs(fichier_apres, preload=True)

    #Sans cet espacement, MNE croit les essais superposés et empile les étiquettes
    avant.events[:, 0] = np.arange(len(avant)) * len(avant.times)
    apres.events[:, 0] = np.arange(len(apres)) * len(apres.times)
    couleurs_avant = {nom: c for nom, c in couleurs_evenements.items() if nom in avant.event_id}
    couleurs_apres = {nom: c for nom, c in couleurs_evenements.items() if nom in apres.event_id}

    avant.plot(title="Sujet " + sujet_id + " - AVANT " + args.Signal, events=True, event_id=True,
               event_color=couleurs_avant, block=False)
    apres.plot(title="Sujet " + sujet_id + " - " + titre_apres, events=True, event_id=True,
               event_color=couleurs_apres, block=True)
