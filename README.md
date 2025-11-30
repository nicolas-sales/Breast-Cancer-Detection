## Breast Cancer Detection – End-to-End Project

This repository contains an end-to-end machine learning project for breast cancer detection using deep learning.
It includes:

Data ingestion, preprocessing, and model training (TensorFlow)

FastAPI backend exposing a prediction endpoint

Streamlit web application

Model storage on AWS S3

Dockerized FastAPI deployment on AWS EC2

Full CI/CD pipeline with GitHub Actions and Amazon ECR


1. Project Structure


breast-cancer-detection/
│
├── app_fastapi.py              # FastAPI backend API (deployed on EC2 + ECR)
├── app_streamlit.py            # Streamlit web UI (deployed on Streamlit Cloud)
├── Dockerfile_fastapi          # Dockerfile for FastAPI app
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── structure.txt               # Notes / quick project structure draft
├── research.ipynb              # Exploration + model training notebook (local dev)
├── best_model.keras            # Local copy of trained model (ignored on GitHub if >100MB)
├── temp_uploaded.jpg           # Temporary image example (dev only)
│
├── .gitignore                  # Git ignore rules
├── .dockerignore               # Files/folders excluded from Docker build context
│
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline (build + push to ECR + deploy on EC2)
│
├── src/
│   ├── logger.py               # Central logging configuration
│   ├── exception.py            # Custom exception class
│   │
│   ├── components/             # Core ML components
│   │   ├── data_ingestion.py       # Load images + split train/val/test
│   │   ├── data_transformation.py  # Image normalization + generators
│   │   ├── model_trainer.py        # CNN model definition + training
│   │   ├── model_evaluation.py     # Metrics, ROC, confusion matrix
│   │   └── prediction.py           # Inference + S3 model download
│   │
│   └── pipeline/               # Orchestrated pipelines
│       ├── data_ingestion_pipeline.py
│       ├── data_transformation_pipeline.py
│       ├── model_trainer_pipeline.py
│       ├── model_evaluation_pipeline.py
│       └── prediction_pipeline.py
│
├── artifacts/                  # Local artifacts (model, metrics, plots, etc.)
│
├── logs/                       # Log files generated during pipelines / API runs
│
├── test_images/                # Sample images used to test prediction pipeline
│
├── docker/                     # (Optional) Docker-related helper files / scripts
│
└── breast_cancer_public_data/  # Raw dataset (Negative / Cancer images)
    └── ...                     # Often ignored or handled as a submodule



2. Model Storage on AWS S3

Since the trained model is larger than 100 MB, it cannot be stored on GitHub.

The model is uploaded to an S3 bucket:

Bucket name: breast-cancer-model-nico

Model key: best_model.keras

The model is downloaded at runtime through environment variables:

S3_BUCKET=breast-cancer-model-nico
S3_MODEL_KEY=best_model.keras


The FastAPI container fetches the model from S3 on startup.

3. FastAPI

The prediction endpoint:

POST /predict


Send an image (JPEG/PNG).
The API returns:

{
  "class": "Cancer",
  "probability": 0.873
}


Swagger documentation is automatically available at:

/docs

4. Docker Deployment (FastAPI)

The FastAPI app is containerized using Dockerfile_fastapi.
The model is not included inside the image; instead, it is downloaded from S3.

To build manually:

docker build -f Dockerfile_fastapi -t fastapi-breast .


To run locally:

docker run -p 8000:8000 fastapi-breast

5. AWS EC2 Deployment with GitHub Actions

A self-hosted GitHub Actions runner is installed on an EC2 instance (t2.medium).
The deployment workflow performs:

Build and push Docker image to Amazon ECR

Connect to EC2 runner

Pull the latest image

Run the container with S3 environment variables

The container is launched using:

docker run -d -p 8000:8000 \
  --ipc="host" \
  --name=breast-fastapi \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_REGION=us-east-1 \
  -e S3_BUCKET=breast-cancer-model-nico \
  -e S3_MODEL_KEY=best_model.keras \
  <ECR_URI>/<REPOSITORY>:latest

6. Streamlit Cloud Deployment

The Streamlit application downloads the model from S3 using Streamlit secrets:

In the Streamlit Cloud dashboard:

Settings → Secrets

Add:

AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
AWS_REGION="us-east-1"
S3_BUCKET="breast-cancer-model-nico"
S3_MODEL_KEY="best_model.keras"


The Streamlit app uses these credentials to pull the model and run inference.

7. Requirements

Key dependencies:

tensorflow
opencv-python-headless
fastapi
uvicorn
boto3
numpy
pillow
streamlit

8. How to Use the API

Example request:

curl -X POST "http://<EC2_PUBLIC_IP>:8000/predict" \
  -F "file=@image.jpg"


Example response:

{
  "class": "Negative",
  "probability": 0.142
}

9. Notes

Ensure port 8000 is open in the EC2 security group.

The instance must stay running for the API to be reachable.

The model must exist in the S3 bucket before deployment.















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

