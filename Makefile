# =============================================================================
# Mini-ML Platform Makefile
# =============================================================================

.PHONY: help up down restart status logs build clean health

help:
	@echo.
	@echo =================================================================
	@echo   Mini-ML Platform
	@echo =================================================================
	@echo.
	@echo Commands:
	@echo   make up        - Start all services
	@echo   make down      - Stop all services
	@echo   make restart   - Restart all services
	@echo   make status    - Show container status
	@echo   make logs      - Show all logs
	@echo   make logs-api  - Show API logs
	@echo   make build     - Rebuild images
	@echo   make clean     - Remove containers and volumes
	@echo   make health    - Check service health
	@echo   make urls      - Show service URLs
	@echo.

up:
	@echo Starting Mini-ML Platform...
	docker compose up -d
	@echo.
	@echo Services started. Run 'make urls' to see all URLs.

down:
	@echo Stopping Mini-ML Platform...
	docker compose down

restart: down up

status:
	docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-inference:
	docker compose logs -f inference

logs-training:
	docker compose logs -f training

build:
	docker compose build --no-cache

clean:
	@echo WARNING: This will delete all data!
	docker compose down -v --remove-orphans

health:
	@echo Checking services...
	@echo.
	@echo API:
	@curl -s http://localhost:8010/health || echo FAIL
	@echo.
	@echo Inference:
	@curl -s http://localhost:8085/health || echo FAIL
	@echo.

urls:
	@echo.
	@echo =================================================================
	@echo   Service URLs
	@echo =================================================================
	@echo.
	@echo   API:        http://localhost:8010
	@echo   Docs:       http://localhost:8010/docs
	@echo   Inference:  http://localhost:8085/health
	@echo   MLflow:     http://localhost:5001
	@echo   MinIO:      http://localhost:9001 [minio/minio12345]
	@echo   Grafana:    http://localhost:3000
	@echo   Prometheus: http://localhost:9090
	@echo   Prefect:    http://localhost:4200
	@echo.

shell:
	docker compose exec api /bin/bash

.DEFAULT_GOAL := help
