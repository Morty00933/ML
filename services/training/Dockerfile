# services/training/Dockerfile
FROM python:3.11

WORKDIR /app

# Ускоряем pip и избегаем сборки из исходников
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ONLY_BINARY=:all:

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8001
CMD ["uvicorn", "flow:app", "--host", "0.0.0.0", "--port", "8001"]
