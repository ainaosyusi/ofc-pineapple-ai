# OFC Pineapple AI - Web Application

ブラウザでOFC Pineapple AIと対戦できるWebアプリケーション。

## ローカル実行

```bash
cd webapp
pip install -r requirements.txt
python app.py
# http://localhost:8000 でアクセス
```

## デプロイ

### Render.com (推奨)

1. GitHubリポジトリを接続
2. `webapp` ディレクトリをRoot Directoryに設定
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Railway

1. GitHubリポジトリを接続
2. `webapp` ディレクトリを選択
3. 自動デプロイ

### Docker

```bash
cd webapp
docker build -t ofc-pineapple-ai .
docker run -p 8000:8000 ofc-pineapple-ai
```

### Heroku

```bash
cd webapp
heroku create ofc-pineapple-ai
git subtree push --prefix webapp heroku main
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | メインページ (HTML) |
| `/api/game/new` | POST | 新規ゲーム作成 |
| `/api/game/place` | POST | カード配置 |
| `/api/game/discard` | POST | カード捨て |
| `/api/game/{id}` | GET | ゲーム状態取得 |
| `/api/health` | GET | ヘルスチェック |

## 技術スタック

- FastAPI
- Uvicorn
- Pure JavaScript (フレームワークなし)
