# Documentation TD1: Prétraitement des données textuelles
Ce document fournit un aperçu et des détails sur le processus de prétraitement des données textuelles pour le projet d'analyse vidéo. TD1 se concentre sur la préparation des données textuelles extraites des noms de vidéos pour les rendre prêtes pour l'analyse de machine learning.

## Description des données
Les données sont structurées avec les colonnes suivantes :

video_name : Le titre de la vidéo.
is_name : Indicateur booléen pour la présence d'un nom propre.
is_comic : Indicateur booléen pour la présence d'un nom de comique.
comic_name : Le nom du comique si présent.
tokens : Les mots tokenisés du titre de la vidéo.

## Processus de prétraitement
Le script make_features.py est utilisé pour transformer les titres textuels des vidéos en caractéristiques numériques utilisables par les modèles de machine learning. Les étapes suivantes sont appliquées :

**Conversion en minuscules** : Tous les textes sont convertis en minuscules pour maintenir la cohérence et éviter la distinction entre les majuscules et les minuscules.
**Suppression des mots vides** : Les mots fréquents qui n'apportent pas de valeur significative pour l'analyse sont retirés.
**Racinisation (Stemming)** : Les mots sont ramenés à leur racine pour réduire la complexité du texte.
**Tokenisation** : Le texte est divisé en tokens (mots ou entités significatives).
**Création de n-grammes** : Des séquences de n mots sont créées pour capturer le contexte local du texte.
**Plage de n-grammes** : Des n-grammes sur une plage de tailles sont générés pour capturer divers niveaux de séquences de mots.
## Utilisation
Pour utiliser ce script de prétraitement, les utilisateurs doivent appeler la fonction make_features(df, task, config) avec les paramètres suivants :

_df_ : DataFrame contenant les données textuelles.
_task_ : La tâche spécifique à réaliser (par exemple, 'is_comic_video').
_config_ : Un dictionnaire de configuration pour activer ou désactiver certaines étapes de prétraitement.

## Experiment

`python main.py evaluate --model_name logistic_regression --task is_comic_video --input_filename data/raw/train.csv `

#### Got accuracy 91.79497487437185%

`python main.py evaluate --model_name random_forest --task is_comic_video --input_filename data/raw/train.csv`

#### Got accuracy 91.69497487437187%

`python main.py evaluate --model_name svm --task is_comic_video --input_filename data/raw/train.csv`

#### Got accuracy 91.8964824120603%

## Tâches de reconnaissance d'entités nommées (NER)

La tâche `is_name` vise à identifier et classer les segments de texte des titres de vidéos qui correspondent à des noms propres. Pour cela, nous utilisons une approche de machine learning qui nécessite les étapes suivantes :

- **Extraction des caractéristiques**: Chaque token du titre de la vidéo est transformé en un vecteur de caractéristiques qui inclut la casse du mot, sa position dans le titre, ainsi que le contexte donné par les tokens précédents et suivants.
- **Étiquetage**: Les tokens sont associés à des étiquettes binaires indiquant la présence (`1`) ou l'absence (`0`) d'un nom propre.

## Tâche de détection de nom de comique

La tâche `find_comic_name` a pour objectif de localiser et extraire les noms de comiques dans les titres de vidéos. Cette tâche peut être traitée comme un problème de classification ou de NER, en fonction de la complexité des données et des noms à identifier.

## Assemblage des modèles

L'assemblage des modèles fait référence à la combinaison des prédictions issues des tâches `is_name` et `find_comic_name` pour améliorer la précision globale. Nous évaluons les performances des modèles assemblés par validation croisée.

## Résolution des erreurs

Durant l'évaluation des modèles, il est fréquent de rencontrer des erreurs liées à la correspondance entre les caractéristiques et les étiquettes. Voici les étapes de dépannage typiques :

- **Vérification de la cohérence des données**: Assurez-vous que chaque token a une étiquette correspondante.
- **Correction des données**: Si les tokens et les étiquettes ne correspondent pas, revoir le script de préparation des données pour corriger les incohérences.
- **Vérification après modification**: Après toute correction, vérifiez que la longueur de la matrice de caractéristiques `X` correspond au nombre d'étiquettes `y`.

## Conclusion

Les expériences montrent que le prétraitement des données et la sélection des caractéristiques ont un impact significatif sur la performance des modèles de machine learning. La précision obtenue dans les tâches `is_comic_video` montre l'efficacité des modèles entraînés et suggère des voies d'amélioration pour les tâches de NER et de détection de noms de comiques.




