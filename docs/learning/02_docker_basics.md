# Docker による ML 開発環境 - 学習ノート

## 概要

Dockerは、アプリケーションをコンテナという独立した環境で実行するための技術です。
機械学習プロジェクトでは、以下の利点があります：

- **環境の再現性**: 「自分のPCでは動く」問題の解消
- **依存関係の分離**: プロジェクトごとに異なるライブラリバージョンを使用可能
- **デプロイの簡素化**: 同じイメージをローカル/クラウドで実行

---

## 🐳 基本概念

### イメージ (Image)
```
コンテナの「設計図」
- OS + ライブラリ + コードをパッケージ化
- Dockerfileから作成
- Docker Hubで公開・共有
```

### コンテナ (Container)
```
イメージから作成される「実行インスタンス」
- 独立したプロセス空間
- ホストOSのカーネルを共有（VMより軽量）
- 停止しても状態を保持可能
```

### ボリューム (Volume)
```
データの永続化領域
- コンテナを削除してもデータが残る
- ホストとコンテナ間でファイル共有
```

---

## 📄 Dockerfile の解説

```dockerfile
# ベースイメージの指定
FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# ファイルのコピー
COPY requirements.txt .

# コマンドの実行（イメージビルド時）
RUN pip install -r requirements.txt

# 環境変数の設定
ENV PYTHONPATH=/app

# デフォルトの実行コマンド
CMD ["python", "train.py"]
```

### 主要なディレクティブ

| ディレクティブ | 説明 | 例 |
|-------------|------|-----|
| `FROM` | ベースイメージ | `python:3.9-slim` |
| `WORKDIR` | 作業ディレクトリ | `/app` |
| `COPY` | ファイルコピー | `COPY . .` |
| `RUN` | ビルド時コマンド | `RUN pip install ...` |
| `ENV` | 環境変数 | `ENV DEBUG=1` |
| `CMD` | デフォルト実行コマンド | `CMD ["python", "app.py"]` |
| `EXPOSE` | ポート公開宣言 | `EXPOSE 8080` |

---

## 🔧 よく使うコマンド

### イメージ操作

```bash
# イメージのビルド
docker build -t myapp:latest .

# イメージ一覧
docker images

# イメージの削除
docker rmi myapp:latest
```

### コンテナ操作

```bash
# コンテナの起動（フォアグラウンド）
docker run myapp

# バックグラウンド起動
docker run -d myapp

# ボリュームマウント付き起動
docker run -v $(pwd)/models:/app/models myapp

# 環境変数付き起動
docker run -e "DEBUG=1" myapp

# コンテナ一覧
docker ps        # 実行中のみ
docker ps -a     # 全て

# コンテナ停止/削除
docker stop <container_id>
docker rm <container_id>

# ログ確認
docker logs -f <container_id>

# コンテナ内に入る
docker exec -it <container_id> bash
```

---

## 📦 Docker Compose

複数のコンテナを管理するためのツール。

### docker-compose.yml の例

```yaml
version: '3.8'

services:
  training:
    build: .
    volumes:
      - ./models:/app/models
    environment:
      - TOTAL_TIMESTEPS=100000
```

### Compose コマンド

```bash
# ビルド
docker-compose build

# 起動
docker-compose up        # フォアグラウンド
docker-compose up -d     # バックグラウンド

# 停止
docker-compose down

# ログ確認
docker-compose logs -f training

# 状態確認
docker-compose ps
```

---

## 🏗️ Multi-stage Build

イメージサイズを小さくするテクニック。

```dockerfile
# ステージ1: ビルド用
FROM python:3.9 as builder
RUN pip install --user torch  # 大量の依存関係

# ステージ2: 実行用（軽量イメージ）
FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
```

**メリット:**
- ビルドツールを最終イメージに含めない
- イメージサイズが数GBから数百MBに削減可能

---

## 💡 OFCプロジェクトでの使用例

### ローカル開発

```bash
# 開発モード（ソースコードをマウント）
docker-compose --profile dev up dev
```

### 本番学習

```bash
# 環境変数を設定してビルド
docker-compose build

# バックグラウンドで学習開始
docker-compose up -d training

# ログを監視
docker-compose logs -f training
```

---

## 🔗 次のステップ

- [x] ローカルでDockerビルドを実行 ✅
- [x] docker-composeで学習を開始 ✅
- [ ] EC2にデプロイ
- [ ] 長時間学習の実行

---

## 📊 テスト実行結果（2026/01/15）

```
Docker build: 3分29秒
学習テスト (5000ステップ):
  - 実行時間: 24.5秒
  - ゲーム数: 8
  - FPS: 290
  - モデル保存: models/ofc_selfplay_*.zip
```
