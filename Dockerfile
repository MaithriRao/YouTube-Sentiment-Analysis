FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# RUN apt-get update && \
#     apt-get install -y libgomp1 && \
#     rm -rf /var/lib/apt/lists/*

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "flask_api.app:app"]

# CMD ["python3", "flask_api/app.py"]