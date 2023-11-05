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




