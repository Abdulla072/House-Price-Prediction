import os
import sys
import json
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from house_price.exception.exception import CustomException
from house_price.logger.logger import logging


class ModelTrainer:

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

        self.metrics_path = os.path.join(
            project_root,
            "artifacts",
            "metrics.json"
        )

    def initiate_model_training(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        try:

            logging.info(
                "Model Training Started"
            )

            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            joblib.dump(
                model,
                self.model_path
            )

            metrics = {
                "MAE": float(mae),
                "R2": float(r2)
            }

            with open(
                self.metrics_path,
                "w"
            ) as file:

                json.dump(
                    metrics,
                    file,
                    indent=4
                )

            logging.info(
                "Model Saved Successfully"
            )

            print(f"MAE: {mae:.2f}")
            print(f"R2 Score: {r2:.4f}")

            return r2

        except Exception as e:

            raise CustomException(
                e,
                sys
            )