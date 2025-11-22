import sys
from src.logger import logging
from src.exception import CustomException

from src.components.prediction import Predict


class PredictionPipeline:
    def __init__ (self,model_path:str,img_path:str):
        self.model_path=model_path
        self.img_path=img_path

    def run(self):
        try:
            logging.info("Starting Prediction pipeline")

            predictor = Predict(model_path=self.model_path)
            pred , prob = predictor.predict(self.img_path)

            result = {
                "prediction" : int(pred),
                "probability" : float(prob)
            }

            logging.info(f"[PredictionPipeline] Done: {result}")

            return result

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":

    model_path = "artifacts/best_model.keras"
    img_path = "test_images/test_0.jpg"

    pipeline = PredictionPipeline(model_path = model_path,img_path = img_path)
    result = pipeline.run()

    print("Prediction Pipeline Successful")
    print(result)

    if result["prediction"] == 0:
        print("Resulat final : Negative")
    else:
        print("résultat final : Cancer")
