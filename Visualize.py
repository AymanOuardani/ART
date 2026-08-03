"""
Ouvre les fenêtres MNE pour regarder un signal à l'œil, avant et après débruitage.

  python Visualize.py brut 1         signal continu, brut puis nettoyé ICLabel (2 fenêtres)
  python Visualize.py pretraite 1    essais prétraités
  python Visualize.py ART 1          avant / après débruitage (2 fenêtres)
  python Visualize.py ART 4 1        avant / après, checkpoint LOSO de l'epoch 1

Signaux : brut · pretraite · ART · ICUNet · ICUNet++ · ICUNet_attn · DuoCL · GCTNet · ICLABEL
Les essais sont colorés par classe (gauche, droite, repos).
"""

import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne

mne.set_log_level("ERROR")

#Chemins des fichiers
Brut_fif = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI")
Brut_fif_ICLABEL = pl.Path(r"C:\Users\aymen\Desktop\ART\Databases\EEGBCI_ICLABEL")
Pretraite = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Prétraité")
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
ART_Epochs = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\ART")

#Nom du fichier débruité selon le modèle (dans Nettoyé/SXXX/)
fichiers_apres = {"ART": "ART.fif",
                  "ICUNet": "ICUNet.fif", "ICUNet++": "ICUNet++.fif", "ICUNet_attn": "ICUNet_attn.fif",
                  "DuoCL": "DuoCL.fif", "GCTNet": "GCTNet.fif",
                  "ICLABEL": "ICLABEL.fif"}   # ICLabel = ICA sur le raw 64 canaux

#Couleurs des évènements (gauche/droite/repos)
couleurs_evenements = {"gauche": "tab:blue", "droite": "tab:red", "repos": "tab:green"}

#Ligne de commande : quoi visualiser + numéro du sujet
signaux = ["brut", "pretraite", *fichiers_apres]
parser = ap.ArgumentParser(description="Visualisation d'un sujet")
parser.add_argument("Signal", help="quoi visualiser (insensible à la casse) : " + ", ".join(signaux))
parser.add_argument("Sujet", type=int, help="Numéro du sujet (1-109)")
parser.add_argument("Epoch", type=int, nargs="?", default=None,
                    help="numéro d'epoch (ART uniquement, ex. 40) -> Output/ART/SXXX/ART_epochN.fif")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

#Résolution insensible à la casse (ex. "art" -> "ART")
correspondance = {nom.lower(): nom for nom in signaux}
if args.Signal.lower() not in correspondance:
    raise SystemExit(f"ERREUR : signal inconnu '{args.Signal}'. Choix possibles : {', '.join(signaux)}")
args.Signal = correspondance[args.Signal.lower()]

if args.Epoch is not None and args.Signal != "ART":
    raise SystemExit("ERREUR : l'argument Epoch n'est utilisable qu'avec le signal ART.")

mne.viz.set_browser_backend("matplotlib")

if args.Signal == "brut":
    #signal continu brut, plus sa version nettoyée par ICLabel si elle existe : 2 fenêtres
    #(évènements T0/T1/T2 nommés et colorés dans les deux)
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
    #epochs prétraités (une fenêtre)
    epochs = mne.read_epochs(Pretraite / (sujet_id + "_Pre.fif"), preload=True)

    #les évènements sauvegardés sont espacés d'1 échantillon (index de bloc, pas un vrai temps) :
    #sans ça MNE les croit tous superposés et empile les étiquettes à l'affichage
    epochs.events[:, 0] = np.arange(len(epochs)) * len(epochs.times)
    couleurs = {nom: c for nom, c in couleurs_evenements.items() if nom in epochs.event_id}

    epochs.plot(title="Sujet " + sujet_id + " - PRÉTRAITÉ", events=True, event_id=True,
                event_color=couleurs, block=True)

else:
    #avant (prétraité) vs après (débruité) en 2 fenêtres séparées
    #tous les modèles nettoyés viennent de Prétraité (3 classes, cf. Clean_Model.py)
    fichier_avant = Pretraite / (sujet_id + "_Pre.fif")
    if args.Epoch is not None:
        fichier_apres = ART_Epochs / sujet_id / f"ART_epoch{args.Epoch}.fif"
        titre_apres = f"APRÈS ART (epoch {args.Epoch})"
    else:
        fichier_apres = Nettoye / sujet_id / fichiers_apres[args.Signal]
        titre_apres = "APRÈS " + args.Signal
    avant = mne.read_epochs(fichier_avant, preload=True)
    apres = mne.read_epochs(fichier_apres, preload=True)

    #les évènements sauvegardés sont espacés d'1 échantillon (index de bloc, pas un vrai temps) :
    #sans ça MNE les croit tous superposés et empile les étiquettes à l'affichage
    avant.events[:, 0] = np.arange(len(avant)) * len(avant.times)
    apres.events[:, 0] = np.arange(len(apres)) * len(apres.times)
    couleurs_avant = {nom: c for nom, c in couleurs_evenements.items() if nom in avant.event_id}
    couleurs_apres = {nom: c for nom, c in couleurs_evenements.items() if nom in apres.event_id}

    avant.plot(title="Sujet " + sujet_id + " - AVANT " + args.Signal, events=True, event_id=True,
               event_color=couleurs_avant, block=False)
    apres.plot(title="Sujet " + sujet_id + " - " + titre_apres, events=True, event_id=True,
               event_color=couleurs_apres, block=True)
