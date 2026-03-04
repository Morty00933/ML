# services/training/Dockerfile
FROM ghcr.io/astral-sh/uv:0.6-python3.11-bookworm-slim
WORKDIR /app

# важно: без "services/training/" и без ведущего слеша
COPY requirements.txt ./requirements.txt
RUN uv pip install --system -r requirements.txt

COPY . ./
EXPOSE 8001
CMD ["uvicorn", "flow:app", "--host", "0.0.0.0", "--port", "8001"]
