from fastapi import FastAPI, UploadFile, File
import os
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Breast Cancer Detection API", description="API for predicting breast cancer from mammogram images")

@app.get("/")
def root():
    return {"message": "Breast Cancer Detection API is running"}

@app.post("/predict", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # --- Correct model path inside container ---
    model_path = "/app/artifacts/best_model.keras"

    pipeline = PredictionPipeline(
        #model_path="artifacts/best_model.keras",
        model_path=model_path,
        img_path=temp_path
    )
    result = pipeline.run()

    os.remove(temp_path)

    label = "Cancer" if result["prediction"] == 1 else "Negative"
    prob = round(result["probability"], 3)

    return {
        "class": label,
        "probability": prob
        }

