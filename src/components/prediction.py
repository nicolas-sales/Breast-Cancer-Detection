import cv2
import boto3
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

from src.logger import logging
from src.exception import CustomException


def download_model_from_s3(local_path: str):
    """
    Télécharge le modèle depuis S3 si non présent localement.
    Utilise les variables d'environnement :
      - S3_BUCKET
      - S3_MODEL_KEY
    """
    try:
        bucket = os.getenv("S3_BUCKET")
        key = os.getenv("S3_MODEL_KEY")

        if not bucket or not key:
            raise CustomException(
                "S3_BUCKET ou S3_MODEL_KEY non définis dans les variables d'environnement",
                sys
            )

        logging.info(f"[Predict] Downloading model from S3: s3://{bucket}/{key}")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3 = boto3.client("s3")
        s3.download_file(bucket, key, local_path)

        logging.info(f"[Predict] Model downloaded to: {local_path}")

    except Exception as e:
        raise CustomException(e, sys)


class Predict:
    def __init__(self, model_path: str):
        try:
            logging.info(f"[Predict] Initializing model loader with path: {model_path}")

            # Vérifier si le modèle existe localement
            if not os.path.exists(model_path):
                logging.info("[Predict] Model not found locally. Downloading from S3...")
                download_model_from_s3(model_path)

            # Charger le modèle une fois disponible
            logging.info(f"[Predict] Loading model from: {model_path}")
            self.model = load_model(model_path)
            logging.info("[Predict] Model loaded successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def preprocess(self, img_path: str):
        try:
            logging.info(f"[Predict] Preprocessing image: {img_path}")

            img = cv2.imread(img_path)
            if img is None:
                raise CustomException(f"Cannot read image: {img_path}", sys)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)  # Shape correct : (1, 224, 224, 3)

            return img

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, img_path: str):
        try:
            img = self.preprocess(img_path)

            prob = float(self.model.predict(img)[0][0])
            pred = 1 if prob > 0.5 else 0

            logging.info(f"[Predict] Prediction: {pred} (prob={prob:.4f})")

            return pred, prob

        except Exception as e:
            raise CustomException(e, sys)
