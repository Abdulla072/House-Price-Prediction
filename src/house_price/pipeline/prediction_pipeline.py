import os
import sys
import joblib
import pandas as pd

from house_price.exception.exception import CustomException


class PredictionPipeline:

    def __init__(self):

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir,
                os.pardir
            )
        )

        self.model_path = os.path.join(
            project_root,
            "artifacts",
            "model.pkl"
        )

        self.preprocessor_path = os.path.join(
            project_root,
            "artifacts",
            "preprocessor.pkl"
        )

    def predict(self, data):

        try:

            model = joblib.load(self.model_path)

            preprocessor = joblib.load(
                self.preprocessor_path
            )

            transformed_data = preprocessor.transform(
                data
            )

            prediction = model.predict(
                transformed_data
            )

            return prediction[0]

        except Exception as e:

            raise CustomException(
                e,
                sys
            )