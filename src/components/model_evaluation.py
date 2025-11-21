import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix,ConfusionMatrixDisplay

from src.logger import logging
from src.exception import CustomException

class ModelEvaluation():
    def __init__(self,model_path: str):
        try:
            logging.info(f"[ModelEvaluation] Loading model from: {model_path}")
            self.model = load_model(model_path)

            # Folder for artifacts
            self.artifact_dir = "artifacts"
            os.makedirs(self.artifact_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)

    # Validation metrics
    def evaluate_validation(self,val_generator,y_val):

        try:
            logging.info("[ModelEvaluation] Predicting on validation set")

            y_pred_proba = self.model.predict(val_generator)
            y_val_pred = (y_pred_proba > 0.5).astype(int)

            precision = precision_score(y_val, y_val_pred)
            recall = recall_score(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred)
            roc = roc_auc_score(y_val, y_pred_proba)

            metrics= {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc)
            }

            # Sauvegarde des métriques
            with open(os.path.join(self.artifact_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info("[ModelEvaluation] Metrics saved to artifacts/metrics.json")

            return y_val,y_val_pred,y_pred_proba
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    # Plot
    def plot_roc(self,y_val,y_pred_proba):
        try:

        
            fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

            auc_score = roc_auc_score(y_val, y_pred_proba)

            plt.figure(figsize=(7,6))
            plt.plot(fpr, tpr, label=f"CNN ROC Curve (AUC = {auc_score:.2f})")

            # Ligne diagonale (hasard)
            plt.plot([0, 1], [0, 1], 'r--')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (1 - Specificity)")
            plt.ylabel("True Positive Rate (Sensitivity)")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True)
            
            out_path = os.path.join(self.artifact_dir,"roc_curve.png")
            plt.savefig(out_path)
            plt.close()

            logging.info(f"[ModelEvaluation] ROC curve saved: {out_path}")

        except Exception as e:
            raise CustomException(e, sys)

    
    def plot_confusion_matrix(self,y_val,y_val_pred):
        try:

            cm = confusion_matrix(y_val, y_val_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Cancer"])

            plt.figure(figsize=(6,6))
            disp.plot(cmap="Blues", values_format="d")
            plt.title("Confusion Matrix")
            
            out_path = os.path.join(self.artifact_dir, "confusion_matrix.png")
            plt.savefig(out_path)
            plt.close()

            logging.info(f"[ModelEvaluation] Confusion matrix saved: {out_path}")

            return cm
        
        except Exception as e:
            raise CustomException(e, sys)