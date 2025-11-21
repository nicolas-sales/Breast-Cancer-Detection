import sys
from src.logger import logging
from src.exception import CustomException

from src.components.model_evaluation import ModelEvaluation
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerPipeline


class ModelEvaluationPipeline:
    def __init__(self, folder_path: str, model_path: str):
        self.folder_path=folder_path
        self.model_path=model_path

    def run(self):
        try:

            logging.info("Starting Model Evaluation Pipeline")

            # Récupération des données
            transformation_pipeline = DataTransformationPipeline(folder_path=self.folder_path)
            train_generator, val_generator, X_test, y_test,y_val = transformation_pipeline.run()

            logging.info("[ModelEvaluationPipeline] Data loaded and transformed")

            # Chargement modèle et validation evaluation

            eval = ModelEvaluation(model_path=self.model_path)

            y_val, y_val_pred, y_val_proba = eval.evaluate_validation(val_generator=val_generator,y_val=y_val)

            eval.plot_roc(y_val,y_val_proba)

            eval.plot_confusion_matrix(y_val,y_val_pred)

            logging.info("Model Evaluation Pipeline Completed")

            return {
                "y_val": y_val,
                "y_val_pred": y_val_pred,
                "y_val_proba": y_val_proba
            }


        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ == "__main__":

    folder = "breast_cancer_public_data/data_2"
    model_path = "artifacts/best_model.keras"  

    pipeline = ModelEvaluationPipeline(folder_path=folder, model_path=model_path)
    results = pipeline.run()

    print("\nModel Evaluation Pipeline Successful")
    print(f"Pred shape : {results['y_val_pred'].shape}")


