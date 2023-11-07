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
    X, y = make_features(df, task, config={"use_lowercase": True, "use_stopwords": True, "use_stemming": True,
                                           "use_tokenization": True, "use_ngram": True, "n_value": 3,
                                           "use_ngram_range": True, "min_n_value": 1, "max_n_value": 4})

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
    X, y = make_features(df, task, config={"use_lowercase": True, "use_stopwords": True, "use_stemming": True,
                                           "use_tokenization": True, "use_ngram": True, "n_value": 3,
                                           "use_ngram_range": True, "min_n_value": 1, "max_n_value": 4})
    model = RandomForestModel()
    model.load(model_dump_filename)
    y_pred = model.predict(X)
    print(y_pred)
    df["prediction"] = y_pred
    df.to_csv(output_filename, index=False)


# command to run: python main.py test --task is_comic_video --input_filename data/raw/train.csv --model_dump_filename models/dump.json --output_filename data/processed/prediction.csv

@click.command()
@click.option("--model_name", default="logistic_regression",
              help="Model to use for evaluation. Can be random_forest, logistic_regression, etc.")
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
# @click.option("--feature", default="lowercase", help="Feature to use for evaluation. Can be lowercase, stopwords, stemming, tokenize, ngram, ngram_range")
def evaluate(model_name, task, input_filename):
    config = {
        # Add your default configurations here, for example:
        "use_lowercase": True,
        # You can add more default configurations as necessary
    }
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    # Diagnostic print statements
    print("Diagnostic Information:")
    print("Number of videos:", len(df))
    print("Total number of tokens:", sum(len(tokens) for tokens in df["tokens"]))
    print("Total number of labels:", sum(len(labels) for labels in df["is_name"]))
    print("Shape of feature matrix X:", X.shape)
    print("Length of label vector y before flattening:", len(y))

    # Flatten y if task is 'is_name' to match the number of tokens
    if task == "is_name":
        y = [label for sublist in df["is_name"] for label in sublist]
        print("Length of label vector y after flattening:", len(y))

    # Check if we still have a mismatch in sample sizes
    if X.shape[0] != len(y):
        raise ValueError(f"Inconsistent numbers of samples: X={X.shape[0]}, y={len(y)}")

    model = make_model(model_name)
    scores = evaluate_model(model, X, y)
    print(f"Got accuracy {100 * np.mean(scores)}%")
    return scores


def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, scoring="accuracy")
    print(f"Got accuracy {100 * np.mean(scores)}%")
    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
