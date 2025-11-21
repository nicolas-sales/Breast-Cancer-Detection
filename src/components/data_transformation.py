import sys
from src.logger import logging
from src.exception import CustomException
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataTransformation:

    def __init__(self):
        logging.info("DataTransformation initialized")

    def initiate_data_transformation(self,X_train,X_val,y_train,y_val):

        try:
            logging.info("Starting Data Tranformation")

            # Train
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=5,
                zoom_range=0.05,
                horizontal_flip=True
            )

            # Val
            val_datagen = ImageDataGenerator(
                rescale=1./255
            )

            # Train Generator
            train_generator = train_datagen.flow(
                X_train,
                y_train,
                batch_size=32,
                shuffle=True
            )

            # Validation Generator
            val_generator = val_datagen.flow(
                X_val,
                y_val,
                batch_size=32,
                shuffle=False
            )

            logging.info("Data Transformation completed")

            return train_generator,val_generator
        
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    folder = "breast_cancer_public_data/data_2"

    # 1. Data Ingestion
    ingestion = DataIngestion(folder)
    X, y = ingestion.load_images()
    X, y = ingestion.shuffle_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = ingestion.split_data(X, y)

    # 2. Data Transformation
    transformation = DataTransformation()
    train_generator, val_generator = transformation.initiate_data_transformation(
        X_train, X_val, y_train, y_val
    )

    print("DataTransformation successful")
    print("Train generator batch shape :", train_generator[0][0].shape)
    print("Val generator batch shape   :", val_generator[0][0].shape)

