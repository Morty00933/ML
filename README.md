# Mini-ML Platform

Мини-облачная ML-платформа (упрощённый аналог AWS SageMaker), построенная на Docker Compose.  
Проект демонстрирует полный ML lifecycle: **от загрузки данных до обучения, деплоя, инференса и мониторинга**.  

---

## Основные возможности

- **Проекты и датасеты**: REST API для создания рабочих пространств и загрузки данных (хранение в MinIO/S3).
- **Обучение моделей**: запуск экспериментов с Optuna, сохранение артефактов и параметров в MLflow.
- **Model Registry**: регистрация моделей и управление стадиями (Staging / Production).
- **Деплой**: REST API для загрузки моделей в отдельный inference-сервис.
- **Инференс**: онлайн-предсказания через FastAPI endpoint.
- **Мониторинг**: Prometheus собирает метрики API и инференса, Grafana визуализирует дашборды.
- **Оркестрация**: Prefect для управления задачами и автопайплайнами.
- **Инфраструктура**: всё разворачивается одной командой через Docker Compose.

---

## Архитектура проекта

```
.
├── services/                # Основные сервисы
│   ├── api/                 # Управление проектами, датасетами, тренировкой и деплоем
│   ├── training/            # Обучение моделей (Optuna + MLflow)
│   └── inference/           # Инференс сервис (FastAPI + MLflow)
│
├── infra/                   # Конфиги инфраструктуры
├── monitoring/              # Prometheus и Grafana дашборды
├── data/samples/            # Пример датасета (iris.csv)
├── docker-compose.yml       # Описание сервисов
├── .env.example             # Пример переменных окружения
└── README.md
```

### Контейнеры

| Сервис      | Порт         | Описание |
|-------------|-------------|----------|
| **API** (FastAPI)      | http://localhost:8000/docs | Управление проектами, датасетами, тренировкой и деплоем |
| **Training** (FastAPI) | http://localhost:8001/docs | Обучение моделей |
| **Inference** (FastAPI) | http://localhost:8080/docs | Предсказания |
| **MLflow**   | http://localhost:5001 | Трекинг экспериментов и Model Registry |
| **MinIO**    | http://localhost:9001 | S3-хранилище артефактов (логин: `minio`, пароль: `minio12345`) |
| **Postgres** | localhost:5432 | База для MLflow |
| **Prefect**  | http://localhost:4200 | Оркестрация задач |
| **Prometheus** | http://localhost:9090 | Сбор метрик |
| **Grafana**  | http://localhost:3000 | Визуализация метрик |

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Morty00933/ML.git
cd ML
```

### 2. Настроить окружение

```bash
cp .env.example .env
```

### 3. Запустить все сервисы

```bash
docker compose up -d --build
```

### 4. Проверить статус

```bash
docker compose ps
```

---

## Пример: обучение и деплой Iris модели

### 1. Создать проект
```bash
curl -X POST http://localhost:8000/projects   -H "X-API-Key: admin-key-123"   -H "Content-Type: application/json"   -d '{"name":"demo"}'
```

### 2. Загрузить датасет
```bash
curl -X POST "http://localhost:8000/datasets/upload?project=demo"   -H "X-API-Key: engineer-key-123"   -F "file=@data/samples/iris.csv"
```

### 3. Запустить обучение
```bash
curl -X POST http://localhost:8000/experiments/train   -H "X-API-Key: engineer-key-123"   -H "Content-Type: application/json"   -d '{
    "project":"demo",
    "dataset_s3_uri":"s3://mlflow/projects/demo/datasets/iris.csv",
    "target":"target",
    "model_name":"demo-iris",
    "test_size":0.2,
    "max_trials":6
  }'
```

### 4. Деплойнуть модель
```bash
curl -X POST http://localhost:8000/deploy   -H "X-API-Key: admin-key-123"   -H "Content-Type: application/json"   -d '{"model_name":"demo-iris","stage":"Staging"}'
```

### 5. Сделать предсказание
```bash
curl -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -d '{"rows":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]}'
```
 Ответ:
```json
{"preds":["Iris-setosa"]}
```

---

## Мониторинг

### Метрики
- API: `http://localhost:8000/metrics`
- Inference: `http://localhost:8080/metrics`

### Grafana дашборды:
- Requests per second
- p95 latency
- Errors per second
- Total requests

### Пример:
- API загрузки датасета → графики запросов и latency  
- Inference → количество предсказаний и ошибки  

---

## Роли и доступ

| Роль       | API Key         | Возможности |
|------------|----------------|-------------|
| **admin**  | `admin-key-123`   | создание проектов, деплой моделей |
| **engineer** | `engineer-key-123` | загрузка датасетов, тренировка моделей |
| **viewer** | `viewer-key-123`   | просмотр метрик |

---

## Roadmap (планы)

- [ ] Поддержка PyTorch и LightGBM моделей  
- [ ] Автоматический retraining (Prefect flow)  
- [ ] CI/CD пайплайн (GitHub Actions → Docker Hub → Kubernetes)  
- [ ] JWT авторизация вместо простых API-ключей  
- [ ] Расширенные метрики (GPU usage, model drift, data drift)  
- [ ] Поддержка multi-tenant (разные рабочие пространства)  

---

## Автор

 [Morty00933](https://github.com/Morty00933)  

Проект создан как pet-project для демонстрации навыков **ML Engineer / AI Infrastructure Developer**:  
- Docker, Docker Compose  
- FastAPI  
- MLflow + Optuna  
- Prefect  
- Prometheus + Grafana  
- MinIO (S3)  
- Postgres  
