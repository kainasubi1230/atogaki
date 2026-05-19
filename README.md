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

## GPUサーバー起動手順

1. サーバー側で NVIDIA ドライバ + nvidia-container-toolkit を導入し、`nvidia-smi` が通る状態にする
2. このリポジトリをサーバーへ配置し、`.env` を設定
3. GPUオーバーライド付きで起動

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

4. GPU認識チェック

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec -T api python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
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

1. 公開データ（Kuzushiji-49）を4万文字取り込み（初回は自動ダウンロード）

```bash
python trainer/main.py import-public-k49 --count 40000 --output storage/base/base_dataset.jsonl --append
```

2. 同意済みかつ前処理済みの成果物からベースデータセット（JSONL）を作成

```bash
python trainer/main.py build-base-dataset --output storage/base/base_dataset.jsonl
```

3. 共有ベースモデルを学習

```bash
python trainer/main.py train-base \
  --dataset storage/base/base_dataset.jsonl \
  --output storage/models/base_model.pt \
  --epochs 20 \
  --batch-size 64 \
  --device cuda
```

4. Docker経由の実行例

```bash
docker compose run --rm trainer python trainer/main.py import-public-k49 --count 40000 --output storage/base/base_dataset.jsonl --append
docker compose run --rm trainer python trainer/main.py build-base-dataset --output storage/base/base_dataset.jsonl
docker compose run --rm trainer python trainer/main.py train-base --dataset storage/base/base_dataset.jsonl --output storage/models/base_model.pt --epochs 20 --batch-size 64 --device cuda
```

注記:
- `POST /generate` は `BASE_MODEL_PATH`（デフォルト: `./storage/models/base_model.pt`）を参照します。
- モデルファイルが存在しない場合はフォールバック生成になります。公開データを使う場合は `train-base` を最低1回実行してください。

## 推論専用モード（GPU PC学習 + 公開サーバー推論）

公開サーバーを推論専用で運用する場合は、以下を設定してください。

- `INFERENCE_ONLY=true`
- `SHARED_STYLE_ID=<共有スタイルID>`
- `BASE_MODEL_PATH=./storage/models/base_model.pt`

この設定時の挙動:
- `POST /styles/{user_id}/train-lora` は学習を実行せず、`SHARED_STYLE_ID` を返します。
- `POST /generate` はリクエストの `style_id` ではなく共有スタイルを使用します。

### セットアップ手順

1. GPU PCで共有スタイルを確認

```bash
python scripts/list_ready_styles.py
```

2. 公開サーバーへアセット同期（例: rsync）

```bash
rsync -avz storage/models/base_model.pt user@SERVER:/opt/machine/storage/models/
rsync -avz storage/adapters/ user@SERVER:/opt/machine/storage/adapters/
```

3. サーバーで `.env` を作成

```bash
cp .env.inference.example .env
# SHARED_STYLE_ID と JWT_SECRET を本番値へ変更
```

4. サーバーDBへ共有スタイルを登録/更新

```bash
python scripts/upsert_shared_style.py --style-id 1 --user-id 1 --adapter-key adapters/u1/style-1-xxxx.json
```

5. 起動

```bash
docker compose up -d --build
```

GPUサーバーの場合:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## ローカルAPIテスト

```bash
python -m pip install -r api/requirements.txt
python -m pytest api/tests -q -p no:cacheprovider
```
