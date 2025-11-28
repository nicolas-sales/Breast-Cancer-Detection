## Breast Cancer Detection – Deep Learning & MLOps Project

- Breast cancer classification from mammography images using a CNN model built with TensorFlow/Keras, deployed through both FastAPI (Docker + AWS) and Streamlit Cloud.

- This project covers the full end-to-end ML pipeline + MLOps deployment:

- Data ingestion

- Data preprocessing & transformation

- Model training (CNN)

- Model evaluation

- Prediction pipeline

- FastAPI production deployment with Docker & AWS (ECR + EC2)

- Streamlit Cloud interactive web app


1. Project Objective

The goal is to classify mammogram images into:

0 – Negative (no cancer)

1 – Cancer detected

The trained model achieves:

- Precision ~0.97

- Recall ~0.95

- F1-score ~0.96

- ROC-AUC ~0.992


2. Project Structure


📁 breast-cancer-detection
│
├── app_fastapi.py
├── app_streamlit.py
│
├── artifacts/
│   └── best_model.keras        
│
├── test_images/
│   ├── test_0.jpg
│   ├── test_1.jpg
│   └── ...
│
├── src/
│   ├── logger.py
│   ├── exception.py
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   ├── prediction.py
│   │
│   ├── pipeline/
│       ├── data_ingestion_pipeline.py
│       ├── data_transformation_pipeline.py
│       ├── model_trainer_pipeline.py
│       ├── model_evaluation_pipeline.py
│       ├── prediction_pipeline.py
│
├── Dockerfile_fastapi
├── .dockerignore
├── requirements.txt
├── README.md
├── .gitignore
│
└── .github/
    └── workflows/
        └── deploy.yml



3. ML Pipeline

- Data Ingestion

Load all images from dataset

Shuffle

Train/Validation/Test split

Output: X_train, X_val, X_test, y_train, y_val, y_test


- Data Transformation

Resize to 224×224

Normalize (pixel / 255)

Data augmentation for training

Generate Keras ImageDataGenerators


- Model Training

Custom CNN architecture

Callbacks:

EarlyStopping

ModelCheckpoint

Saves the best weights to artifacts/best_model.keras


- Model Evaluation

Computes:

Precision

Recall

F1-score

ROC-AUC

Confusion matrix

ROC curve visualization


Outputs saved to:

artifacts/metrics.json
artifacts/roc_curve.png
artifacts/confusion_matrix.png


Prediction Pipeline

Input: an uploaded mammogram
Output:

{
  "class": "Negative",
  "probability": 0.044
}


- 4. FastAPI Deployment (Docker + AWS)
Dockerfile_fastapi includes:

base Python 3.10

system dependencies for OpenCV

application code

automatic download of the Keras model from S3

Uvicorn server setup

Build locally:

docker build -f Dockerfile_fastapi -t breast-fastapi .

Run locally:

docker run -p 8000:8000 breast-fastapi

Visit the FastAPI documentation:
http://localhost:8000/docs


5. CI/CD with GitHub Actions + AWS (ECR + EC2)

Pipeline steps:

- CI (Continuous Integration)

Fetch repository

Lint

Run tests

- CD – Build & Push to Amazon ECR

Build Docker image

Tag image

Push to ECR registry

- Deployment on EC2 (Self-hosted GitHub Runner)

Pull latest Docker image

Run container on EC2

FastAPI becomes publicly accessible

- Large model handling

The file best_model.keras is large.
It is stored in AWS S3 and automatically downloaded at container startup.

Environment variables required:

AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
S3_BUCKET
S3_KEY


6. Streamlit App (Deployed on Streamlit Cloud)

The Streamlit app provides:

Image upload

Mammogram preview

Real-time prediction using the same model

Automatic model download from S3

Run locally:

streamlit run app_streamlit.py

Public app URL:

https://breast-cancer-detection-hcsunwfbr6uqqwhmq6krxm.streamlit.app/


7. Example Prediction Output

Class : Negative
Probability : 0.044


8. Local Installation

Install dependencies:
pip install -r requirements.txt

Run FastAPI:
uvicorn app_fastapi:app --reload

Run Streamlit:
streamlit run app_streamlit.py


9. Technologies Used

Python 3.10

TensorFlow / Keras

OpenCV

scikit-learn

FastAPI

Streamlit Cloud

Docker

AWS S3 / ECR / EC2

GitHub Actions (CI/CD)


10. Conclusion

This project showcases:

complete ML pipeline engineering

model packaging and production deployment

Docker image management

CI/CD with GitHub Actions

cloud deployment (AWS + Streamlit Cloud)

An excellent demonstration of both Deep Learning and MLOps skills.








Fast API:
uvicorn app_fastapi:app --reload
http://127.0.0.1:8000/docs

Streamlit:
streamlit run app_streamlit.py
http://localhost:8501

Docker:
docker build -t breast-fastapi -f Dockerfile_fastapi .
docker run -p 8000:8000 breast-fastapi
http://localhost:8000/docs


EC2:

Installation et demmarge Docker puis ajout de l'utilisateur:
sudo apt-get update -y
sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

ou:

Installation Docker:
Connection SSH à l'instance EC2
sudo apt update
sudo apt install -y docker.io

Demarrage de Docker
sudo systemctl start docker
sudo systemctl enable docker

Ajouter l'utilisateur au groupe docker
sudo usermod -aG docker $USER

Se deconnecter du serveur:
exit

Depuis le pc:
ssh -i "nicolas.pem" ubuntu@<IP_PUBLIC_EC2>  Remplacer <IP_PUBLIC_EC2> par l’IP publique affichée dans EC2

Vérifier que Docker fonctionne sans sudo:
docker ps

