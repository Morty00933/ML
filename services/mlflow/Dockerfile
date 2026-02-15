# services/mlflow/Dockerfile
FROM python:3.11-slim

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем mlflow + драйвер для Postgres
RUN pip install --no-cache-dir mlflow==2.14.1 psycopg2-binary==2.9.9

EXPOSE 5000

# Команда запуска — аргументы передадим из compose
ENTRYPOINT ["mlflow", "server"]
