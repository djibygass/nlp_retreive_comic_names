from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import FrenchStemmer
import nltk
from nltk.corpus import stopwords

# Download the stopwords
nltk.download('stopwords')
# french_stopwords = set(stopwords.words('french'))
french_stopwords = list(stopwords.words('french'))


stemmer = FrenchStemmer()


def tokenize(text):
    tokens = text.split()  # split the text into tokens
    stems = [stemmer.stem(item) for item in tokens]  # stem each token
    return stems


def make_features(df, task):
    y = get_output(df, task)
    X = df["video_name"]

    # vectorizer = CountVectorizer(stop_words=french_stopwords, tokenizer=tokenize)
    # X = vectorizer.fit_transform(X)

    return X, y


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y
