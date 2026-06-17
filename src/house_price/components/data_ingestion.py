import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from house_price.exception.exception import CustomException
from house_price.logger.logger import logging


class DataIngestion:

    def __init__(self):

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir,
                os.pardir
            )
        )

        self.source_data_path = os.path.join(
            project_root,
            "data",
            "train.csv"
        )

        self.artifacts_dir = os.path.join(
            project_root,
            "artifacts"
        )

        self.raw_data_path = os.path.join(
            self.artifacts_dir,
            "raw.csv"
        )

        self.train_data_path = os.path.join(
            self.artifacts_dir,
            "train.csv"
        )

        self.test_data_path = os.path.join(
            self.artifacts_dir,
            "test.csv"
        )

    def initiate_data_ingestion(self):

        logging.info("Data Ingestion Started")

        try:

            print("PROJECT ROOT:", os.path.dirname(self.artifacts_dir))
            print("DATA PATH:", self.source_data_path)

            if not os.path.exists(self.source_data_path):
                raise FileNotFoundError(
                    f"Source data file not found: {self.source_data_path}"
                )

            if os.path.getsize(self.source_data_path) == 0:
                raise ValueError(
                    f"Source data file is empty: {self.source_data_path}"
                )

            df = pd.read_csv(
                self.source_data_path
            )

            os.makedirs(
                self.artifacts_dir,
                exist_ok=True
            )

            df.to_csv(
                self.raw_data_path,
                index=False
            )

            logging.info(
                "Raw dataset saved"
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.train_data_path,
                index=False
            )

            test_set.to_csv(
                self.test_data_path,
                index=False
            )

            logging.info(
                "Train Test Split Completed"
            )

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )