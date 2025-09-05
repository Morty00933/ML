FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ONLY_BINARY=:all:

COPY services/training/requirements.txt ./requirements.txt

# Без апгрейда pip
RUN pip install --no-cache-dir -r requirements.txt

COPY services/training/ ./
EXPOSE 8001
CMD ["uvicorn", "flow:app", "--host", "0.0.0.0", "--port", "8001"]
