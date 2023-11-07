import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

# Download necessary NLTK resources if not already present
nltk.download('punkt')


# Assume this is your French POS tagger function
def french_pos_tag(tokens):
    # This function should return a list of POS tags for the French tokens
    # You would replace this with the actual call to your French POS tagger
    return ['POS_TAG' for _ in tokens]


def make_features(df, task, config=None):
    if config is None:
        config = {}

    if task == "is_name":
        # Extract features for each token in each video name
        features_per_video = df["tokens"].apply(lambda x: extract_features_for_ner(x, config))

        # Flatten the list of feature dictionaries for all videos
        # This step is wrong and should be removed: X = [feature for sublist in X for feature in sublist]

        dict_vectorizer = DictVectorizer()

        # Fit transform on the list of feature dictionaries for all videos
        # This will create a sparse matrix X where each row corresponds to a token's features
        X = dict_vectorizer.fit_transform([item for sublist in features_per_video for item in sublist])
    else:
        # For other tasks, use a simple bag-of-words model for now
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df["video_name"])

    y = get_output(df, task)

    return X, y


def extract_features_for_ner(tokens, config):
    """Extract features for named entity recognition for each token."""
    if config is None:
        config = {}

    pos_tags = french_pos_tag(tokens) if config.get("use_pos_tagging") else [None] * len(tokens)
    context_window = config.get("context_window_size", 1)

    # Features for all tokens
    all_features = []

    for i, token in enumerate(tokens):
        token_features = {
            "token": token.lower(),
            "is_capitalized": token[0].isupper(),
            "is_first": i == 0,
            "is_last": i == (len(tokens) - 1),
            "pos_tag": pos_tags[i],
        }

        # Add context features
        for j in range(1, context_window + 1):
            # Look back
            token_features[f'prev_token_{j}'] = tokens[i - j].lower() if i - j >= 0 else None
            token_features[f'prev_tag_{j}'] = pos_tags[i - j] if i - j >= 0 else None

            # Look forward
            token_features[f'next_token_{j}'] = tokens[i + j].lower() if i + j < len(tokens) else None
            token_features[f'next_tag_{j}'] = pos_tags[i + j] if i + j < len(tokens) else None

        all_features.append(token_features)

    return all_features


def get_output(df, task):
    if task == "is_comic_video":
        return df["is_comic"]
    elif task == "is_name":
        # Flatten all label lists for all tokens in all videos into a single list
        return [label for sublist in df["is_name"] for label in sublist]
    elif task == "find_comic_name":
        return df["comic_name"]
    else:
        raise ValueError("Unknown task")



# ... rest of your existing functions ...


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
