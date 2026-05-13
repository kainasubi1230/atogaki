# アクセシビリティ手書き生成 MVP

このリポジトリは、アクセシビリティ用途に限定した手書き生成MVPです。

## 安全ガードレール

- 生成は `purpose=accessibility` の場合のみ許可
- すべてのSVGに可視ウォーターマークを埋め込み
- APIリクエストと生成イベントの監査ログを保存
- 学習に使うのは同意（consent）済みデータのみ
- スタイル所有者が一致しない場合は `403` を返却

## サービス構成

- `api`: FastAPI
- `worker`: 前処理ジョブ用 RQ Worker
- `trainer`: LoRA学習ジョブ用 RQ Worker
- `web`: Next.js + React
- `postgres`, `redis`, `minio`

## クイックスタート

```bash
docker compose up --build
```

アクセス先:
- API: `http://localhost:8000`
- Web: `http://localhost:3000`
- MinIO Console: `http://localhost:9001`

## API（MVP）

- `POST /auth/signup`
- `POST /auth/login`
- `POST /datasets/upload-scan`
- `POST /datasets/{user_id}/preprocess`
- `POST /styles/{user_id}/train-lora`
- `POST /generate`
- `GET /jobs/{job_id}`
- `GET /outputs/{output_id}`
- `GET /audit/logs?user_id=&from=&to=`

## ベースモデル学習（転移学習準備）

1. 同意済みかつ前処理済みの成果物からベースデータセット（JSONL）を作成

```bash
python trainer/main.py build-base-dataset --output storage/base/base_dataset.jsonl
```

2. 共有ベースモデルを学習

```bash
python trainer/main.py train-base \
  --dataset storage/base/base_dataset.jsonl \
  --output storage/models/base_model.pt \
  --epochs 20 \
  --batch-size 64 \
  --device cuda
```

3. Docker経由の実行例

```bash
docker compose run --rm trainer python trainer/main.py build-base-dataset --output storage/base/base_dataset.jsonl
docker compose run --rm trainer python trainer/main.py train-base --dataset storage/base/base_dataset.jsonl --output storage/models/base_model.pt --epochs 20 --batch-size 64 --device cuda
```

## ローカルAPIテスト

```bash
python -m pip install -r api/requirements.txt
python -m pytest api/tests -q -p no:cacheprovider
```
