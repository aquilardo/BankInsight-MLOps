FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==3.1.1 \
    "sqlalchemy>=2.0,<2.1" \
    psycopg2-binary==2.9.10 \
    boto3==1.38.46

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]