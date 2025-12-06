# ================================================================
# Makefile for Semantic Search Project
# ================================================================

# Variables
IMAGE_NAME = semantic-search
TAG = latest

PYTHON = python
UVICORN = uvicorn app.main:app --host 0.0.0.0 --port 8000

# ================================================================
# Local development commands
# ================================================================

.PHONY: install
install:
	pip install --upgrade pip
	pip install -e .

.PHONY: run
run:
	$(UVICORN)

.PHONY: embed
embed:
	DISABLE_MLFLOW=1 $(PYTHON) -m pipeline.embedder

.PHONY: test
test:
	pytest -q

# ================================================================
# Docker commands
# ================================================================

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

.PHONY: docker-run
docker-run:
	docker run --rm -p 8000:8000 $(IMAGE_NAME):$(TAG)

.PHONY: docker-shell
docker-shell:
	docker run -it --rm $(IMAGE_NAME):$(TAG) /bin/bash

# ================================================================
# Utility commands
# ================================================================

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf build dist *.egg-info
	rm -f data/models/*.npy data/models/*.json

.PHONY: format
format:
	black .
	isort .

