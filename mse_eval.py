import argparse as ap
import pathlib as pl
import numpy as np
import mne as mne
import Utils

mne.set_log_level("ERROR")

#Chemins des fichiers
Nettoye = pl.Path(r"C:\Users\aymen\Desktop\ART\Output\Nettoyé")
Rapport_Tex = pl.Path(r"C:\Users\aymen\Desktop\ART\Résultats\Rapport_ART.tex")

#Libellé de chaque comparaison dans le tableau du rapport (Résultats/Rapport_ART.pdf)
labels_rapport = {"ICA": "ICA vs ICLabel (référence)", "ART": "ART vs ICLabel"}

#Ligne de commande : quel signal comparer à ICLabel (+ numéro du sujet)
parser = ap.ArgumentParser(description="MSE entre un signal nettoyé et le signal nettoyé par ICLabel")
parser.add_argument("Method", choices=list(labels_rapport), help="signal à comparer à ICLabel")
parser.add_argument("Sujet", type=int, help="Numéro du sujet (1-109)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

f_methode = Nettoye / sujet_id / (args.Method + ".fif")
f_iclabel = Nettoye / sujet_id / "ICLABEL.fif"
if not (f_methode.exists() and f_iclabel.exists()):
    raise SystemExit(f"ERREUR : fichier(s) manquant(s) pour {sujet_id}\n"
                     f"  {f_methode} (existe : {f_methode.exists()})\n"
                     f"  {f_iclabel} (existe : {f_iclabel.exists()})")

#Les deux viennent de Prétraité_Total (3 classes) : on retire le repos des deux côtés
methode = mne.read_epochs(f_methode, preload=True)["gauche", "droite"]
iclabel = mne.read_epochs(f_iclabel, preload=True)["gauche", "droite"]

if not np.array_equal(methode.events[:, 2], iclabel.events[:, 2]):
    raise SystemExit(f"ERREUR : {sujet_id} - essais {args.Method} et ICLabel non alignés (labels différents)")

mse = float(np.mean((methode.get_data() - iclabel.get_data()) ** 2))
print(f"{sujet_id} : MSE ({args.Method} vs ICLabel) = {mse:.4e}")

#Reporte la valeur dans Résultats/Rapport_ART.pdf
if Rapport_Tex.exists():
    ligne = f"{labels_rapport[args.Method]} & {mse:.2e} \\\\"
    Utils.maj_tableau_tex(Rapport_Tex.parent / "data" / f"{sujet_id}_mse.json",
                          Rapport_Tex.parent / "data" / f"{sujet_id}_mse.tex",
                          f"{args.Method}_vs_ICLABEL", ligne, macro="mserow",
                          ordre=[f"{m}_vs_ICLABEL" for m in labels_rapport])
    Utils.recompile_latex(Rapport_Tex)
