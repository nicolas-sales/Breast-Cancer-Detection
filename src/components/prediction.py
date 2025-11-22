import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

from src.logger import logging
from src.exception import CustomException


class Predict:
    def __init__(self,model_path:str):
        try:
            logging.info(f"[Predict] Loading model from: {model_path}")
            self.model=load_model(model_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def preprocess(self, img_path:str):

        try:
            logging.info(f"[Predict] Preprocessing image: {img_path}")

            img = cv2.imread(img_path)
            if img is None:
                 raise CustomException(f"Cannot read image: {img_path}", sys)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(224,224))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img,axis=0) # Shape -> (1,224,224,3)

            return img
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self,img_path:str):

        try:
            img = self.preprocess(img_path)

            prob = self.model.predict(img)[0][0]
            pred = 1 if prob > 0.5 else 0

            logging.info(f"[Predict] Prediction: {pred} (prob={prob:.4f})")

            return pred,prob
        
        except Exception as e:
            raise CustomException(e, sys)