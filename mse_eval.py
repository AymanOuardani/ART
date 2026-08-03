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
#ICLabel est toujours la référence (repos retiré, mêmes essais gauche/droite)
labels_rapport = {"ART": "ART", "ICUNet": "ICUNet", "ICUNet++": "ICUNet++", "ICUNet_attn": "ICUNet\\_attn",
                  "DuoCL": "DuoCL", "GCTNet": "GCTNet", "ICA": "ICA"}

#Ligne de commande : quel signal comparer à ICLabel (+ numéro du sujet)
parser = ap.ArgumentParser(description="MSE entre un signal nettoyé et le signal nettoyé par ICLabel")
parser.add_argument("Method", help="signal à comparer à ICLabel (insensible à la casse) : " + ", ".join(labels_rapport))
parser.add_argument("Sujet", type=int, help="Numéro du sujet (1-109)")
parser.add_argument("--sans-latex", action="store_true",
                    help="met à jour les données du rapport sans lancer pdflatex (traitement en série)")
args = parser.parse_args()
sujet_id = "S" + str(args.Sujet).zfill(3)

#Résolution insensible à la casse (ex. "duocl" -> "DuoCL")
correspondance = {nom.lower(): nom for nom in labels_rapport}
if args.Method.lower() not in correspondance:
    raise SystemExit(f"ERREUR : signal inconnu '{args.Method}'. Choix possibles : {', '.join(labels_rapport)}")
args.Method = correspondance[args.Method.lower()]

#ICLabel = l'ICA ICLabel appliquée au signal continu à 64 canaux (ICLABEL_Brut.py), qui est aussi
#la cible d'entraînement d'ART : c'est la seule référence ICLabel du rapport.
f_methode = Nettoye / sujet_id / (args.Method + ".fif")
f_iclabel = Nettoye / sujet_id / "ICLABEL_Amélioré.fif"
if not (f_methode.exists() and f_iclabel.exists()):
    raise SystemExit(f"ERREUR : fichier(s) manquant(s) pour {sujet_id}\n"
                     f"  {f_methode} (existe : {f_methode.exists()})\n"
                     f"  {f_iclabel} (existe : {f_iclabel.exists()})")

#Les deux viennent de Prétraité (3 classes) : tout le signal, comme pendant l'entraînement
#(Train_ART.py n'exclut pas le repos), pas seulement gauche/droite
methode = mne.read_epochs(f_methode, preload=True)
iclabel = mne.read_epochs(f_iclabel, preload=True)

if not np.array_equal(methode.events[:, 2], iclabel.events[:, 2]):
    raise SystemExit(f"ERREUR : {sujet_id} - essais {args.Method} et ICLabel non alignés (labels différents)")

mse = float(np.mean((methode.get_data() - iclabel.get_data()) ** 2))
rmse = np.sqrt(mse) * 1e6   # V -> µV
rms_methode = np.sqrt(np.mean(methode.get_data() ** 2)) * 1e6
ref_rms = np.sqrt(np.mean(iclabel.get_data() ** 2)) * 1e6
print(f"{sujet_id} : RMSE ({args.Method} vs ICLabel) = {rmse:.2f} µV "
      f"(RMS {args.Method} = {rms_methode:.2f} µV, RMS ICLabel = {ref_rms:.2f} µV)")

#Reporte la valeur dans Résultats/Rapport_ART.pdf
if Rapport_Tex.exists():
    ligne = f"{labels_rapport[args.Method]} & {rmse:.2f} & {rms_methode:.2f} & \\\\"
    Utils.maj_tableau_tex(Rapport_Tex.parent / "data" / f"{sujet_id}_mse.json",
                          Rapport_Tex.parent / "data" / f"{sujet_id}_mse.tex",
                          f"{args.Method}_vs_ICLABEL", ligne, macro="mserow",
                          ordre=[f"{m}_vs_ICLABEL" for m in labels_rapport])

    #Colonne RMS_ICLABEL : une seule valeur, centrée sur toutes les lignes du tableau (\multirow)
    tex_path = Rapport_Tex.parent / "data" / f"{sujet_id}_mse.tex"
    contenu = tex_path.read_text(encoding="utf-8")
    lignes = contenu.split("\\def\\mserow{", 1)[1].rstrip("}\n").split("\n")
    n = len(lignes)
    lignes[0] = lignes[0].rsplit("&", 1)[0] + f"& \\multirow{{{n}}}{{*}}{{{ref_rms:.1f}}} \\\\"
    tex_path.write_text("\\def\\mserow{\n" + "\n".join(lignes) + "}\n", encoding="utf-8", newline="\n")
    if not args.sans_latex:
        Utils.recompile_latex(Rapport_Tex)
