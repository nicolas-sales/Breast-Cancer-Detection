import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self,folder_path:str):
        self.folder_path = folder_path

    def run(self):
        try:
            logging.info("Starting Data Ingestion Pipeline")

            ingestion = DataIngestion(folder_path=self.folder_path)

            # Load
            X,y = ingestion.load_images()

            # Shuffle
            X,y = ingestion.shuffle_data(X,y)

            # Split
            X_train, X_val, X_test, y_train, y_val, y_test = ingestion.split_data(X,y)

            logging.info("Data Ingestion Pipeline Completed")

            return {
                "X_train": X_train,
                "X_val" : X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test
            }
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    pipeline = DataIngestionPipeline(
        folder_path="breast_cancer_public_data/data_2"
    )

    data = pipeline.run()

    print("Data Ingestion successful")
    print(type(data["X_train"]), data["X_train"].shape)
    print(type(data["X_val"]),   data["X_val"].shape)
    print(type(data["X_test"]),  data["X_test"].shape)
    print(type(data["y_train"]), data["y_train"].shape)
    print(type(data["y_val"]),   data["y_val"].shape)
    print(type(data["y_test"]),  data["y_test"].shape)