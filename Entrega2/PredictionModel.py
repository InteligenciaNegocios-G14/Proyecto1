from DataPreprocessing import TextPreprocessor, TextVectorizer
from joblib import load
from DataModel import DataModel  # Mantén esto si lo necesitas
import joblib
import DataPreprocessing
pipeline = load('./assets/pipeline_model.joblib', globals=globals())

class Model:
    def __init__(self, columns):
        self.model = load("./assets/pipeline_model.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result