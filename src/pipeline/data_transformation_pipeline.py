import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline


class DataTransformationPipeline:
    def __init__(self,folder_path : str):
        self.folder_path = folder_path

    def run(self):
        try:
            logging.info("Starting Data Transformation Pipeline")

            # Ingestion
            ingestion_pipeline = DataIngestionPipeline(folder_path=self.folder_path)

            data = ingestion_pipeline.run()

            X_train = data["X_train"]
            X_val   = data["X_val"]
            X_test  = data["X_test"]
            y_train = data["y_train"]
            y_val   = data["y_val"]
            y_test  = data["y_test"]

            logging.info("[DataTransformationPipeline] Data successfully ingested.")
            logging.info(f" Train: {X_train.shape}")
            logging.info(f" Val  : {X_val.shape}")
            logging.info(f" Test : {X_test.shape}")

            # Transformation
            transformation = DataTransformation()

            train_generator,val_generator=transformation.initiate_data_transformation(
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val
            )

            logging.info("Data Transformation completed")

            return train_generator,val_generator,X_test,y_test,y_val
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ =="__main__":

    folder = "breast_cancer_public_data/data_2"

    pipeline=DataTransformationPipeline(folder_path=folder)
    train_generator,val_generator,X_test,y_test,y_val = pipeline.run()

    print("DataTransformation Pipeline successful")
    print(" Train batch :", train_generator[0][0].shape, train_generator[0][1].shape)
    print(" Val batch   :", val_generator[0][0].shape, val_generator[0][1].shape)
    print(" X_test      :", X_test.shape)
    print(" y_test      :", y_test.shape)