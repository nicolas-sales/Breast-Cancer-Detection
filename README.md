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
