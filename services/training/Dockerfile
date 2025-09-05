# services/training/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Ускоряем pip и избегаем сборки из исходников
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ONLY_BINARY=:all:

# Сначала зависимости
COPY services/training/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Потом исходники
COPY services/training/ ./

EXPOSE 8001
CMD ["uvicorn", "flow:app", "--host", "0.0.0.0", "--port", "8001"]
