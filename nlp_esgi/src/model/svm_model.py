from sklearn.svm import SVC
import joblib


class SVMModel:
    """
    This class represents a Support Vector Machine (SVM) classifier.
    SVMs are supervised learning models used for classification and regression analysis.
    They work by mapping training data to points in space so as to separate the classes
    into two distinct regions separated by a clear gap that is as wide as possible.
    """

    def __init__(self):
        self.model = SVC()

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
