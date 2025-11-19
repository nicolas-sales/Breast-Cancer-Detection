import os
import cv2
import numpy as np
import sys
from sklearn.model_selection import train_test_split

#ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#sys.path.append(ROOT_DIR)

from src.logger import logging
from src.exception import CustomException
import sys


class DataIngestion:
    def __init__(self,folder_path:str):

        # folder_path : chemin du dossier contenant /Negative et /Cancer

        self.folder_path=folder_path
        self.classes=['Negative','Cancer']

    def load_images(self):

        # Charge les images, les convertit en RGB et les redimensionne

        try:
            logging.info("[DataIngestion] Loading dataset from: {self.folkder_path}")

            X = []
            y = []

            for class_label in self.classes:
                class_path=os.path.join(self.folder_path,class_label)

                if not os.path.exists(class_path):
                    raise CustomException(f"Folder not found : {class_path}", sys)
                
                label_index=self.classes.index(class_label)
                logging.info(f"[DataIngestion] Loading class: {class_label}")

                for img_file in os.listdir(class_path):
                    img_path=os.path.join(class_path, img_file)
                    img = cv2.imread(img_path)

                    if img is None:
                        logging.warning(f"[DataIngestion] Unreadable image: {img_path}")
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
                    img = cv2.resize(img, (224, 224))

                    X.append(img)
                    y.append(label_index)

            X = np.array(X)
            y = np.array(y)

            logging.info(f"[DataIngestion] Total images loaded: {len(X)}")
            return X,y
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def shuffle_data(self,X,y):
        try:
            logging.info("[DataIngestion] Shuffling dataset")

            indices = np.arange(len(X)) # Attribu un indice à chaque images
            np.random.shuffle(indices)  # Mélange les indices

            return X[indices],y[indices] # X et y mélangés de la même façon donc on conserve les couples image/label
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_data(self,X,y):
        try:
            logging.info("[DataIngestion] Splitting dataset")

            X_train,X_temp,y_train,y_temp = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

            X_val,X_test,y_val,y_test = train_test_split(X_temp,y_temp,test_size=0.5,stratify=y_temp,random_state=42)

            logging.info("[DataIngestion] Split completed")
            return X_train, X_val, X_test, y_train, y_val, y_test
        
            logging.info("[DataIngestion] Split completed")
            logging.info(f"Train: {len(X_train)} images")
            logging.info(f"Val:   {len(X_val)} images")
            logging.info(f"Test:  {len(X_test)} images")
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    ingestion = DataIngestion(folder_path="breast_cancer_public_data/data_2")

    X,y = ingestion.load_images()
    X,y = ingestion.shuffle_data(X,y)
    splits = ingestion.split_data(X,y)

    print("Data Ingestion successful")

    for part in splits:
        print(type(part), part.shape if hasattr(part, "shape") else len(part))
        
    