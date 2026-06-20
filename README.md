# あとがき - アクセシビリティ手書き生成

このリポジトリは、アクセシビリティ用途に限定した手書き生成プロダクトです。

入力テキストを単に手書き風フォントへ置き換えるのではなく、文字ごとの軌跡データセットから SVG path を再構成し、読みやすさを保ったまま筆跡らしさを生成します。

詳細な技術説明は [TECHNICAL_OVERVIEW.md](./TECHNICAL_OVERVIEW.md) を参照してください。

## 技術ハイライト

- **フォント置換ではなく軌跡生成**: 文字をストローク列として扱い、SVG path として出力します。
- **データセット駆動**: ひらがな、カタカナ、常用漢字、英数字、記号の JSONL 軌跡データをランタイムで選択します。
- **品質ゲート付きランダム生成**: 候補選択、回転、せん断、線幅、払いをランダム化しつつ、崩れた線はスコアリングで排除します。
- **ユーザー筆跡プロファイル**: アップロードされた手書きサンプルから、線幅、傾き、縦横比、揺れを推定します。
- **編集可能な出力**: 生成結果は画像ではなく SVG path なので、web 上で文字・ストローク単位に編集できます。
- **安全設計**: アクセシビリティ用途制限、可視ウォーターマーク、同意済みデータのみ学習、監査ログを実装しています。

## 現在の生成方式

```text
入力テキスト
  -> 文字種判定
  -> JSONL の軌跡候補を読み込み
  -> 品質スコアで候補選抜
  -> 筆跡ゆらぎとサイズ補正
  -> SVG path としてキャンバスへ配置
```

将来的には、文字単体ではなく `文字列 -> 連続軌跡` として生成することで、字間・サイズ・流れをさらに自然にする方針です。

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

## Web をローカル `npm run dev` で動かす

1. Node.js 22 を有効化（`nvm` 利用）

```bash
nvm install
nvm use
```

2. 依存をインストール

```bash
npm install
```

3. API は Docker で起動したまま、Web コンテナだけ停止（ポート競合回避）

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml stop web
```

4. Web 開発サーバー起動

```bash
npm run dev
```

アクセス先:
- Web: `http://localhost:3000`
- API: `http://localhost:8000`

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
- `POST /datasets/upload-trajectory`
- `POST /datasets/{user_id}/preprocess`
- `POST /styles/{user_id}/train-lora`
- `POST /generate`
- `GET /jobs/{job_id}`
- `GET /outputs/{output_id}`
- `GET /audit/logs?user_id=&from=&to=`

### Web手書き軌跡の収集（学習用）

`Scan` 画面の「Web手書きサンプル（学習データ用）」で描画して「軌跡を保存」を押すと、
`POST /datasets/upload-trajectory` へ軌跡が送信されます。

- 保存時点で `preprocess_status=done` のデータとして登録されるため、`build-base-dataset` で直接学習データに取り込まれます。
- `label` には描いた文字（例: `あ`）を入れてください。

軌跡データだけで学習データセットを作る例:

```bash
docker compose exec -T trainer python trainer/main.py build-base-dataset \
  --output storage/base/base_dataset_trajectories_only.jsonl \
  --object-key-prefix trajectories/
```

その後の再学習:

```bash
docker compose exec -T trainer python trainer/main.py train-base \
  --dataset storage/base/base_dataset_trajectories_only.jsonl \
  --output storage/models/base_model_trajectories_only.pt \
  --epochs 20 \
  --batch-size 128 \
  --device cuda
```

### 1枚画像でかな一括学習

`Scan` 画面で「かな一括ラベルを使う（1枚画像向け）」をONにすると、  
`/datasets/upload-scan` に複数文字ラベル（`labels`）を渡します。

- 画像内の文字は **左上から読み順** で切り出して対応付けます。
- 例: ひらがな→カタカナの順で並べて1枚に書いた画像をアップロード。
- そのまま `train-lora` まで自動で進むので、ユーザー別フォント学習に使えます。

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

## 英字・数字データセット

英語入力用の `A-Z` / `a-z` / `0-9` 軌跡データを生成:

```bash
python trainer/main.py build-latin-dataset \
  --output storage/base/base_dataset_latin_v1.jsonl \
  --variants-per-char 48 \
  --replace
```

Docker/GPU 構成で再生成する場合:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm trainer \
  python trainer/main.py build-latin-dataset \
    --output storage/base/base_dataset_latin_v1.jsonl \
    --variants-per-char 48 \
    --replace
```

`api` は `storage/base/base_dataset_latin_v1.jsonl` をランタイム描画に読み込みます。base model にも英字 exemplar を入れたい場合は、この JSONL を学習用データに結合してから `train-base` を実行してください。

## 常用漢字フル対応データセット（KanjiVG）

常用漢字 2136 字をまとめて取り込む:

```bash
python trainer/main.py import-kanjivg-joyo \
  --output storage/base/base_dataset_kanjivg_joyo_full_v5.jsonl \
  --variants-per-char 3 \
  --compact-scale 0.90 \
  --hand-jitter 1.20 \
  --replace \
  --source-name kanjivg-joyo-v5
```

ユーザー軌跡データ（`base_dataset_from_trajectory.jsonl`）と統合して学習用を作る例:

```bash
cat storage/base/base_dataset_kanjivg_joyo_full_v5.jsonl \
    storage/base/base_dataset_from_trajectory.jsonl \
  > storage/base/base_dataset_joyo_full_plus_user_v1.jsonl
```

そのまま再学習:

```bash
python trainer/main.py train-base \
  --dataset storage/base/base_dataset_joyo_full_plus_user_v1.jsonl \
  --output storage/models/base_model_joyo_full_plus_user_v1.pt \
  --epochs 40 \
  --batch-size 128 \
  --device cuda
```

運用時は `.env` の `BASE_MODEL_PATH` を上記出力に合わせてください。

## ユーザー別フォント（転移学習）のポイント

- `POST /datasets/upload-trajectory` で **1文字ラベル付き**（例: `漢`）の軌跡を保存すると、`train-lora` で文字別 exemplar と筆跡プロファイルを作成します。
- `POST /styles/{user_id}/train-lora` は、同意済み/前処理済みデータからユーザー固有アダプタを生成します。
- `POST /generate` は、ベースモデル exemplar に加えてユーザー exemplar を優先利用し、未学習文字はベースへフォールバックします。

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
