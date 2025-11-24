import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import CustomException


model_path = "artifacts/best_model.keras"

st.set_page_config(page_title="Breast Cancer Detection", layout="centered") # titre de l’onglet du navigateur

st.title("Breast Cancer Detection from Mammogram")
st.write("Upload a mammogram image and the model will predict if it shows cancer or not")

uploaded_file=st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Affichage de l'image
    img = Image.open(uploaded_file)
    st.image(img, caption="Image uploaded", use_container_width=True)

    # Sauvegarde temporaire
    temp_path = "temp_uploaded.jpg"
    img.save(temp_path)

    if st.button("Predict"):

        try:
            pipeline = PredictionPipeline(model_path=model_path, img_path=temp_path)
            result = pipeline.run()

            pred=result["prediction"]
            prob=result["probability"]

            label = "Negative" if pred==0 else "Cancer"

            st.subheader("Result")
            st.write(f"Prediction : {label}")
            st.write(f"Probability : '{prob:.3f}'")

        except Exception as e:
            st.error(f"Error during prediction: {e}")