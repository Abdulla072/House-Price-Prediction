from house_price.components.data_ingestion import DataIngestion
from house_price.components.data_transformation import DataTransformation
from house_price.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    transformer = DataTransformation()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = transformer.initiate_data_transformation(
        train_path,
        test_path
    )

    trainer = ModelTrainer()

    trainer.initiate_model_training(
        X_train,
        X_test,
        y_train,
        y_test
    )
    