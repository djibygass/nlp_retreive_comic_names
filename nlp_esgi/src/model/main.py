from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def make_model(model_name):
    """
    Create a machine learning pipeline based on the given model name.

    Args:
    - model_name (str): The name of the machine learning model to use.
                        Can be one of ["random_forest", "logistic_regression", "svm"].

    Returns:
    - pipeline: A scikit-learn pipeline with the specified model.
    """

    if model_name == "random_forest":
        classifier = RandomForestClassifier()
    elif model_name == "logistic_regression":
        classifier = LogisticRegression()
    elif model_name == "svm":
        classifier = SVC()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        (model_name, classifier)
    ])
