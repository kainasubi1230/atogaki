# Accessibility Handwriting MVP

This repository provides an accessibility-focused handwriting generation MVP.

Safety guardrails included:
- Generation is allowed only with `purpose=accessibility`
- Every SVG includes a visible watermark: `AI生成（アクセシビリティ支援）`
- Audit logs are stored for API requests and generation events
- Only consented user data is allowed in training
- Style ownership mismatch is rejected with `403`

## Services

- `api`: FastAPI
- `worker`: RQ worker for preprocessing jobs
- `trainer`: RQ worker for LoRA training jobs
- `web`: Next.js + React
- `postgres`, `redis`, `minio`

## Quick Start

```bash
docker compose up --build
```

Endpoints:
- API: `http://localhost:8000`
- Web: `http://localhost:3000`
- MinIO Console: `http://localhost:9001`

## API (MVP)

- `POST /auth/signup`
- `POST /auth/login`
- `POST /datasets/upload-scan`
- `POST /datasets/{user_id}/preprocess`
- `POST /styles/{user_id}/train-lora`
- `POST /generate`
- `GET /jobs/{job_id}`
- `GET /outputs/{output_id}`
- `GET /audit/logs?user_id=&from=&to=`

## Base Model Training (Transfer Learning Preparation)

1. Build base dataset JSONL from consented preprocessed artifacts:

```bash
python trainer/main.py build-base-dataset --output storage/base/base_dataset.jsonl
```

2. Train the shared base model:

```bash
python trainer/main.py train-base \
  --dataset storage/base/base_dataset.jsonl \
  --output storage/models/base_model.pt \
  --epochs 20 \
  --batch-size 64 \
  --device cuda
```

3. Docker execution example:

```bash
docker compose run --rm trainer python trainer/main.py build-base-dataset --output storage/base/base_dataset.jsonl
docker compose run --rm trainer python trainer/main.py train-base --dataset storage/base/base_dataset.jsonl --output storage/models/base_model.pt --epochs 20 --batch-size 64 --device cuda
```

## Local API Test

```bash
python -m pip install -r api/requirements.txt
python -m pytest api/tests -q -p no:cacheprovider
```

