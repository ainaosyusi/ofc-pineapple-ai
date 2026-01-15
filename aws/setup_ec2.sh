#!/bin/bash
# ===========================================
# OFC Pineapple AI - EC2 Setup Script
# ===========================================
# 使用方法:
#   1. EC2インスタンスを起動 (Amazon Linux 2 or Ubuntu 22.04推奨)
#   2. このスクリプトを実行: bash setup_ec2.sh
#   3. 環境変数を設定: vim /app/.env
#   4. 学習開始: bash run_training.sh

set -e

echo "=============================================="
echo "OFC Pineapple AI - EC2 Setup"
echo "=============================================="

# -------------------------------------------
# System Updates
# -------------------------------------------
echo "[1/6] Updating system..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    sudo apt-get update -y
    sudo apt-get install -y git curl wget htop
elif command -v yum &> /dev/null; then
    # Amazon Linux
    sudo yum update -y
    sudo yum install -y git curl wget htop
fi

# -------------------------------------------
# Docker Installation
# -------------------------------------------
echo "[2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "Docker installed. Please log out and log back in for group changes."
else
    echo "Docker already installed."
fi

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# -------------------------------------------
# Docker Compose Installation
# -------------------------------------------
echo "[3/6] Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    echo "Docker Compose already installed."
fi

# -------------------------------------------
# AWS CLI Installation
# -------------------------------------------
echo "[4/6] Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws
else
    echo "AWS CLI already installed."
fi

# -------------------------------------------
# Project Setup
# -------------------------------------------
echo "[5/6] Setting up project..."
APP_DIR="/app/ofc-training"
sudo mkdir -p $APP_DIR
sudo chown -R $USER:$USER /app

# Clone or update repository (リポジトリURLを設定してください)
# git clone https://github.com/YOUR_REPO/ofc-pineapple-ai.git $APP_DIR
# cd $APP_DIR

# 手動でファイルをコピーする場合:
echo "Please copy project files to $APP_DIR"

# -------------------------------------------
# Environment Setup
# -------------------------------------------
echo "[6/6] Setting up environment..."

if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$APP_DIR/.env.example" ]; then
        cp "$APP_DIR/.env.example" "$APP_DIR/.env"
        echo "Created .env from .env.example. Please edit it with your settings."
    fi
fi

# -------------------------------------------
# Helper Scripts
# -------------------------------------------
cat > /usr/local/bin/ofc-start << 'EOF'
#!/bin/bash
cd /app/ofc-training
docker-compose up -d training
docker-compose logs -f training
EOF
sudo chmod +x /usr/local/bin/ofc-start

cat > /usr/local/bin/ofc-stop << 'EOF'
#!/bin/bash
cd /app/ofc-training
docker-compose down
EOF
sudo chmod +x /usr/local/bin/ofc-stop

cat > /usr/local/bin/ofc-status << 'EOF'
#!/bin/bash
cd /app/ofc-training
docker-compose ps
docker-compose logs --tail=50 training
EOF
sudo chmod +x /usr/local/bin/ofc-status

# -------------------------------------------
# Summary
# -------------------------------------------
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Copy project files to $APP_DIR"
echo "  2. Edit .env file: vim $APP_DIR/.env"
echo "  3. Build Docker image: cd $APP_DIR && docker-compose build"
echo "  4. Start training: ofc-start"
echo ""
echo "Available commands:"
echo "  ofc-start  - Start training"
echo "  ofc-stop   - Stop training"
echo "  ofc-status - Check status and logs"
echo ""
