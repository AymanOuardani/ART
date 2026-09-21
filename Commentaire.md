# Style des commentaires et vocabulaire du projet

Mémo pour garder les scripts homogènes.

---

## Vocabulaire

Les noms des modèles et des signaux sont figés. Ne pas en inventer d'autres, ne pas les
traduire, ne pas les paraphraser.

| Terme | Ce qu'il désigne |
|---|---|
| **ART_Orig** | le modèle ART de l'article, poids d'origine, jamais réentraîné |
| **ART_Local** | notre ART, réentraîné sur EEGBCI contre ICLabel |
| **ICLabel** | l'ICA décomposée sur le signal continu à 64 canaux, dont ICLabel trie les composantes. Il n'y en a qu'un seul dans le projet |
| **brut** | le signal tel qu'enregistré, seulement prétraité |
| **essai** | un bloc de 4 s, jamais « epoch » (réservé aux epochs d'entraînement) |
| **epoch** | une passe d'entraînement, ou le checkpoint qui en sort |

Ces noms sont aussi ceux du code et des fichiers : `Model/ART_Orig/`, `Model/ART_Local/`,
`Output/ART_Local/`, `Output/Nettoyé/SXXX/ART_Orig.fif`. Un seul nom par objet, partout.

**À ne jamais écrire** : « LOSO », « leave-one-subject-out ». Dire « un modèle par sujet
exclu » ou simplement « par sujet ».

---

## Écriture

`#` collé au texte, pas d'espace : `#Chemins des fichiers`.
Les commentaires en fin de ligne gardent leur espace : `WIN = 512   # fenêtre mono-canal`.

Français, sans abréviation. Une ligne. Deux au maximum, et seulement quand la deuxième
évite une erreur.

---

## Quoi commenter

Un commentaire annonce **ce que fait le bloc qui suit**, pas comment il le fait — la lecture
du code donne déjà le comment.

```python
#Lecture des essais prétraités du sujet
#Débruitage essai par essai
#Sauvegarde, la sortie est toujours réécrite
```

On garde une explication **uniquement quand elle protège d'une erreur** : un choix
contre-intuitif, un piège de bibliothèque, une contrainte physique.

```python
#Le décodeur reçoit le signal bruité, jamais la cible : sinon il apprendrait
#à la recopier, ce qu'il ne pourra pas faire à l'inférence.

#Sans cet espacement, MNE croit les essais superposés et empile les étiquettes
```

Ne pas commenter ce que le nom dit déjà. `#Ligne de commande` au-dessus du bloc `argparse`
suffit ; inutile d'y détailler chaque argument, `--help` s'en charge.

---

## En-tête de fichier

Chaque script s'ouvre sur une docstring en trois parties : ce que fait le script et pourquoi,
les lignes de commande, puis les précisions utiles.

```python
"""
Mesure ce qui reste de l'intention motrice dans un signal : décodage main gauche vs main
droite par CSP + LDA, en validation croisée.

  python Evaluation.py brut 1      un sujet
  python Evaluation.py ART         les 109 sujets, avec la moyenne finale
  python Evaluation.py ART 4 1     ART au checkpoint de l'epoch 1, sujet 4

Toutes les méthodes portent sur exactement les mêmes essais : le repos est retiré juste
avant la classification.
"""
```

La première phrase dit **à quoi sert le script**, pas ce qu'il contient. « Mesure ce qui
reste de l'intention motrice » plutôt que « Script d'évaluation CSP+LDA ».

`Utils.py` porte la mention qu'il ne se lance pas directement.

---

## Ligne de commande

Arguments positionnels uniquement, jamais d'options en `--`. Ce qui se règle rarement va
en constante en tête de fichier, avec un commentaire d'une ligne.

```python
#Sujets à traiter : None = tous, ou une liste
sujets_a_traiter = None
```

Ordre des arguments, identique partout : `Signal Sujet [Epoch]`.

---

## Noms de variables

Jamais de majuscules pleines. Une constante s'écrit comme le reste, première lettre en
capitale et mots séparés par un tiret bas : `Modeles`, `Win`, `Seuil_brain`, `Model_Dir`,
`Art_Template`. Pas de `MODELES` ni de `WIN`.

Les variables locales restent en minuscules : `sujet_id`, `data_clean`, `rmse_train`.

---

## Code

Déroulé linéaire, de haut en bas, sans définir de fonction quand une seule suffit. Une
fonction seulement si elle évite une vraie duplication.

Les blocs sont séparés par une ligne vide et introduits par leur commentaire, ce qui rend
le fichier lisible d'un coup d'œil sans avoir à sauter d'une définition à l'autre.
