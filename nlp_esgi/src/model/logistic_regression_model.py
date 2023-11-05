from sklearn.linear_model import LogisticRegression
import joblib


class LogisticRegressionModel:
    """
    This class represents a Logistic Regression model.
    It's a simple linear classifier used for binary classification tasks.
    """

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        """Fit the model to the given training data."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict the class labels for the provided data."""
        return self.model.predict(X)

    def dump(self, filename_output):
        """Serialize and save the model to the specified file."""
        joblib.dump(self.model, filename_output)

    def load(self, filename_input):
        """Load and deserialize the model from the specified file."""
        self.model = joblib.load(filename_input)
