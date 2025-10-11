FROM python:3.12

WORKDIR /app

COPY . /app

# COPY lgbm_model.pkl .

# COPY tfidf_vectorizer.pkl .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "flask_api/app.py"]