from sklearn.ensemble import RandomForestClassifier
import joblib


class RandomForestModel:
    """
    This class represents a Random Forest classifier.
    Random forests are an ensemble learning method for classification,
    regression and other tasks that operate by constructing a multitude of
    decision trees during training and outputting the class that is the
    mode of the classes of the individual trees for classification.
    """

    def __init__(self):
        self.model = RandomForestClassifier()

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
