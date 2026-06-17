import os
import sys
import joblib

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from house_price.exception.exception import CustomException
from house_price.logger.logger import logging


class DataTransformation:

    def __init__(self):

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir,
                os.pardir
            )
        )

        self.preprocessor_path = os.path.join(
            project_root,
            "artifacts",
            "preprocessor.pkl"
        )

    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                "LotFrontage",
                "LotArea",
                "OverallQual",
                "OverallCond",
                "YearBuilt",
                "YearRemodAdd",
                "MasVnrArea",
                "BsmtFinSF1",
                "BsmtUnfSF",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
                "GrLivArea",
                "FullBath",
                "GarageCars",
                "GarageArea"
            ]

            categorical_columns = [
                "MSZoning",
                "Street",
                "LotShape",
                "Neighborhood"
            ]

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "SalePrice"

            X_train = train_df.drop(
                columns=[target_column]
            )

            y_train = train_df[target_column]

            X_test = test_df.drop(
                columns=[target_column]
            )

            y_test = test_df[target_column]

            preprocessor = (
                self.get_data_transformer_object()
            )

            X_train_arr = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_test_arr = (
                preprocessor.transform(
                    X_test
                )
            )

            os.makedirs(
                os.path.dirname(
                    self.preprocessor_path
                ),
                exist_ok=True
            )

            joblib.dump(
                preprocessor,
                self.preprocessor_path
            )

            logging.info(
                "Preprocessor Saved"
            )

            return (
                X_train_arr,
                X_test_arr,
                y_train,
                y_test
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )