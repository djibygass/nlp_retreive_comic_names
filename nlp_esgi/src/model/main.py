from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
# from src.model.dumb_model import DumbModel
from sklearn.linear_model import LogisticRegression


def make_model(model_type="random_forest"):
    if model_type == "random_forest":
        classifier = RandomForestClassifier()
    elif model_type == "logistic_regression":
        classifier = LogisticRegression()
    else:
        raise ValueError("Unknown model type")

    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        (model_type, classifier),
    ])
