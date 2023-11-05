import nltk
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


# Téléchargement des ressources nécessaires s'ils ne sont pas déjà présents
# nltk.download('stopwords')
# nltk.download('punkt')

def make_features(df, task, config=None):
    """Convertit le nom des vidéos en vecteurs de fonctionnalités en utilisant différentes méthodes de prétraitement.

    Args:
        df (DataFrame): DataFrame contenant les noms des vidéos et les étiquettes.
        task (str): La tâche pour laquelle extraire les étiquettes (ex: "is_comic_video").
        config (dict, optional): Configuration pour le prétraitement. Par défaut à None.

    Returns:
        X, y: Matrices des fonctionnalités et des étiquettes.
    """

    #X = df["video_name"]
    X = df["tokens"].apply(' '.join)
    y = get_output(df, task)

    if task == "is_name":
        pass


    if config:
        if config.get("use_lowercase", False):
            X = X.str.lower()
        if config.get("use_stopwords", False):
            X = X.apply(remove_stopwords)
        if config.get("use_stemming", False):
            X = X.apply(stemming)
        if config.get("use_tokenization", False):
            X = X.apply(tokenize)
        if config.get("use_ngram", False):
            n_val = config.get("n_value", 3)
            X = X.apply(lambda text: make_ngrams(text, n=n_val))
        if config.get("use_ngram_range", False):
            min_val = config.get("min_n_value", 1)
            max_val = config.get("max_n_value", 4)
            X = X.apply(lambda text: make_ngrams_range(text, min_n=min_val, max_n=max_val))

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)

    return X, y


def get_output(df, task):
    """Récupère les étiquettes correspondantes à la tâche spécifiée."""

    if task == "is_comic_video":
        return df["is_comic"]
    elif task == "is_name":
        return df["is_name"]
    elif task == "find_comic_name":
        return df["comic_name"]
    else:
        raise ValueError("Unknown task")


def remove_stopwords(text):
    """Supprime les mots vides du texte donné."""

    stops = set(stopwords.words('french'))
    return " ".join([word for word in word_tokenize(text) if word not in stops])


def stemming(text):
    """Applique la racinisation sur le texte donné."""

    stemmer = SnowballStemmer(language='french')
    return " ".join([stemmer.stem(word) for word in word_tokenize(text)])


def tokenize(text):
    """Tokenise le texte donné."""

    return " ".join(word_tokenize(text))


def make_ngrams(text, n=3):
    """Crée des n-grammes à partir du texte donné."""

    return ' '.join([' '.join(grams) for grams in ngrams(word_tokenize(text), n)])


def make_ngrams_range(text, min_n=1, max_n=4):
    """Crée des n-grammes de différentes tailles à partir du texte donné."""

    words = word_tokenize(text)
    n_grams = []
    for n in range(min_n, max_n + 1):
        n_grams += [' '.join(grams) for grams in ngrams(words, n)]
    return ' '.join(n_grams)
