"""Main script for the project.
Entry point for trainning, testing and evaluating the model based on the task.
You can run this script from the command line by typing:
    python src/main.py train --task is_comic_video
    python src/main.py test --task is_comic_video
    python src/main.py evaluate --task is_comic_video
"""

import click
import numpy as np
from sklearn.model_selection import cross_val_score
from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main import make_model
from model.dumb_model import DumbModel
from model.random_forest_model import RandomForestModel
from model.logistic_regression_model import LogisticRegressionModel
from model.svm_model import SVMModel

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task, config={"use_lowercase": True, "use_stopwords": True, "use_stemming": True, "use_tokenization": True, "use_ngram": True, "n_value": 3, "use_ngram_range": True, "min_n_value": 1, "max_n_value": 4})

    model = RandomForestModel()
    model.fit(X, y)

    return model.dump(model_dump_filename)

# command to run: python main.py train --task is_comic_video --input_filename data/raw/train.csv --model_dump_filename models

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task, config={"use_lowercase": True, "use_stopwords": True, "use_stemming": True, "use_tokenization": True, "use_ngram": True, "n_value": 3, "use_ngram_range": True, "min_n_value": 1, "max_n_value": 4})
    model = RandomForestModel()
    model.load(model_dump_filename)
    y_pred = model.predict(X)
    print(y_pred)
    df["prediction"] = y_pred
    df.to_csv(output_filename, index=False)

# command to run: python main.py test --task is_comic_video --input_filename data/raw/train.csv --model_dump_filename models/dump.json --output_filename data/processed/prediction.csv

@click.command()
@click.option("--model_name", default="logistic_regression", help="Model to use for evaluation. Can be random_forest, logistic_regression, etc.")
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
#@click.option("--feature", default="lowercase", help="Feature to use for evaluation. Can be lowercase, stopwords, stemming, tokenize, ngram, ngram_range")
def evaluate(model_name, task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)
    feature_config = {
        "use_lowercase": False,  # Proper names are often capitalized, so case may be important
        "use_stopwords": False,  # Stop words are unlikely to be useful for this task
        "use_stemming": False,  # Stemming may not be appropriate as proper names do not require stemming
        "use_tokenization": True,  # Tokenization is essential to process individual words
        "use_pos_tagging": True,  # Part-of-speech tags can be very helpful to identify proper names
        "context_window_size": 2,  # You may want to look at a few words before and after
        "use_ngram": False,  # N-grams are likely less useful for single word entity recognition
        "use_capitalization_feature": True,  # Capitalization is a strong indicator of proper names
        # Additional features specific to NER can be added here
    }

    if task == "is_name":
        # Update the feature_config specifically for 'is_name' task
        feature_config.update({
            "use_capitalization_feature": True,
            "context_window_size": 2,  # Context size of 2 could mean 2 words before and after the current word
            "use_pos_tagging": True,  # Assuming you have a POS tagger set up for French
            # Other features for named-entity recognition can be added here
        })

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task, config=feature_config)

    # Object with .fit, .predict methods
    model = make_model(model_name)

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)

# command to run: python main.py evaluate --model_name logistic_regression --task is_comic_video --input_filename data/raw/train.csv


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
