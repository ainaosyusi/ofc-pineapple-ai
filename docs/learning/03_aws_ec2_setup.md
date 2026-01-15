# AWS EC2 でのML学習 - 学習ノート

## 概要

AWS EC2（Elastic Compute Cloud）はクラウド上の仮想サーバーです。
機械学習では、ローカルPCより高性能なインスタンスで長時間学習を実行できます。

---

## 🏗️ EC2の基本概念

### インスタンスタイプ

| カテゴリ | タイプ例 | 用途 |
|---------|---------|------|
| **汎用** | t3.xlarge | 開発・テスト |
| **コンピューティング最適化** | c6a.2xlarge | CPU学習 |
| **GPU** | g4dn.xlarge | GPU学習 |

### 料金体系

| 種類 | 説明 | コスト |
|------|------|--------|
| **オンデマンド** | 使った分だけ支払い | 基準価格 |
| **スポット** | 余剰キャパシティを利用 | 最大90%OFF |
| **リザーブド** | 1-3年契約で割引 | 最大75%OFF |

> **💡 ヒント:** 学習には**スポットインスタンス**がおすすめ。
> 中断リスクはあるが、チェックポイント保存で対応可能。

---

## 🔧 セットアップ手順

### 1. EC2インスタンスの起動

**AWS Console操作:**
1. EC2 ダッシュボード → 「インスタンスを起動」
2. AMI: **Ubuntu 22.04 LTS** を選択
3. インスタンスタイプ: **t3.xlarge** または **c6a.2xlarge**
4. キーペア: 新規作成または既存を選択
5. セキュリティグループ: SSH (22) を許可
6. ストレージ: 30GB以上推奨
7. 「起動」をクリック

### 2. SSH接続

```bash
# キーファイルの権限設定
chmod 400 your-key.pem

# SSH接続
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 3. 初期セットアップ

```bash
# プロジェクトファイルをアップロード
scp -i your-key.pem -r ./OFC\ NN ubuntu@<EC2_PUBLIC_IP>:~/

# または git clone
git clone https://github.com/YOUR_REPO/ofc-pineapple-ai.git

# セットアップスクリプト実行
cd ofc-pineapple-ai
bash aws/setup_ec2.sh
```

---

## 📦 Docker でのデプロイ

```bash
# 環境変数設定
cp .env.example .env
vim .env  # Webhook URL等を設定

# イメージビルド
docker-compose build

# 学習開始（バックグラウンド）
docker-compose up -d training

# ログ監視
docker-compose logs -f training

# 状態確認
docker-compose ps
```

---

## 💰 コスト最適化

### スポットインスタンスの使用

```bash
# AWS CLI でスポットインスタンスをリクエスト
aws ec2 request-spot-instances \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

### 自動停止設定

```bash
# 学習完了後に自動シャットダウン
# .env に AUTO_SHUTDOWN=true を設定

# または手動で設定
sudo shutdown -h +60  # 60分後にシャットダウン
```

---

## 🔔 進捗通知の設定

### Discord Webhookの取得

1. Discordサーバーの設定を開く
2. 「連携サービス」→「ウェブフック」
3. 「新しいウェブフック」を作成
4. URLをコピーして `.env` に設定

### .env 設定例

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxxx/yyyyy
TOTAL_TIMESTEPS=1000000
NOTIFICATION_INTERVAL=50000
```

---

## 📊 S3へのチェックポイント保存

```bash
# AWS CLI 設定
aws configure

# S3バケット作成
aws s3 mb s3://ofc-training-bucket

# 手動アップロード
aws s3 sync ./models/ s3://ofc-training-bucket/models/
```

---

## 🔗 便利なコマンド

```bash
# インスタンスにファイルをコピー
scp -i key.pem local_file ubuntu@IP:~/remote_file

# インスタンスからファイルをダウンロード
scp -i key.pem ubuntu@IP:~/models/*.zip ./

# 実行中のインスタンス一覧
aws ec2 describe-instances --query 'Reservations[].Instances[?State.Name==`running`]'

# インスタンスの停止
aws ec2 stop-instances --instance-ids i-xxxxx

# インスタンスの終了（削除）
aws ec2 terminate-instances --instance-ids i-xxxxx
```

---

## ⚠️ 注意事項

1. **インスタンスを停止するのを忘れない！** 
   - 実行中は課金され続ける
   - 不要になったら必ず停止/終了

2. **セキュリティグループの設定**
   - SSH (22) のみ許可
   - 必要最小限のIPからのみアクセス許可

3. **スポットインスタンスの中断対策**
   - 定期的にチェックポイントを保存
   - S3に自動アップロード設定

---

## 🔗 次のステップ

- [ ] EC2インスタンス起動
- [ ] プロジェクトファイルのアップロード
- [ ] 長時間学習の開始
- [ ] 結果の確認とモデルのダウンロード
