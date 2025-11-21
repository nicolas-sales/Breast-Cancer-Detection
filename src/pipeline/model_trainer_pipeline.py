import sys
from src.logger import logging
from src.exception import CustomException

from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.components.model_trainer import ModelTrainer


class ModelTrainerPipeline():
    def __init__(self,folder_path):
        self.folder_path=folder_path

    def run(self):
        try:
            logging.info("Starting Model training pipeline")

            # Transformation
            transformation_pipeline = DataTransformationPipeline(folder_path=self.folder_path)
            train_generator,val_generator,X_test,y_test, _ = transformation_pipeline.run()  # ne prend pas la dernière valeur "y_val"

            logging.info("[ModelTrainerPipeline] Data transformation OK")

            # Training
            trainer=ModelTrainer()
            model, checkpoint_path, history = trainer.train(train_generator,val_generator)

            logging.info("Model trainer pipeline completed")
            logging.info(f"Best model saved at: {checkpoint_path}")

            return model, checkpoint_path, X_test, y_test
        
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":

    folder = "breast_cancer_public_data/data_2"

    pipeline = ModelTrainerPipeline(folder_path=folder)

    model, checkpoint_path, X_test, y_test = pipeline.run()

    print("\nModel Trainer Pipeline Successful")
    print("Best model :", checkpoint_path)
    print("Test set   :", X_test.shape, y_test.shape)

