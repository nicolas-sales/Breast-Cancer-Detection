Breast cancer detection

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

sudo apt-get update -y
sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

ou:

Installatioin Docker:
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

