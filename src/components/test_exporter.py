import os
import cv2
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys


class TestSetExporter:

    def __init__(self, output_folder="test_images"):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def save_test_set(self, X_test, y_test):
        
        try:
            logging.info("[TestSetExporter] Saving test images")

            labels_list = []

            for i in range(len(X_test)):
                img = X_test[i]

                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                filename = f"test_{i}.jpg"
                filepath = os.path.join(self.output_folder, filename)

                cv2.imwrite(filepath, img_bgr)

                labels_list.append({
                    "filename": filename,
                    "label": int(y_test[i])
                })

            # Création CSV
            df = pd.DataFrame(labels_list)
            csv_path = os.path.join(self.output_folder, "test_labels.csv")
            df.to_csv(csv_path, index=False)

            logging.info(f"[TestSetExporter] Saved {len(X_test)} images.")
            logging.info(f"[TestSetExporter] Labels saved: {csv_path}")

            return self.output_folder, csv_path

        except Exception as e:
            raise CustomException(e, sys)