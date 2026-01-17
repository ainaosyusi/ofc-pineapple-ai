#!/bin/bash
# ===========================================
# OFC Pineapple AI - GCE Setup Script
# ===========================================
# 使用方法:
#   1. GCEインスタンスを起動 (Ubuntu 22.04 LTS推奨、n2-standard-4以上)
#   2. このスクリプトを実行: bash setup_gce.sh
#   3. 環境変数を設定: vim /app/.env
#   4. 学習開始: bash run_training.sh
#
# 推奨スペック:
#   - CPU: 4+ vCPU (n2-standard-4 以上)
#   - メモリ: 16GB+
#   - ディスク: 50GB+ SSD
#   - GPU: オプション (NVIDIA T4推奨)

set -e

echo "=============================================="
echo "OFC Pineapple AI - GCE Setup"
echo "=============================================="

# -------------------------------------------
# System Updates
# -------------------------------------------
echo "[1/7] Updating system..."
sudo apt-get update -y
sudo apt-get install -y git curl wget htop unzip software-properties-common

# -------------------------------------------
# Python 3.9 Installation
# -------------------------------------------
echo "[2/7] Installing Python 3.9..."
if ! command -v python3.9 &> /dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.9 python3.9-venv python3.9-dev python3-pip
fi

# pip upgrade
python3.9 -m pip install --upgrade pip

# -------------------------------------------
# Docker Installation
# -------------------------------------------
echo "[3/7] Installing Docker..."
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
echo "[4/7] Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    echo "Docker Compose already installed."
fi

# -------------------------------------------
# Google Cloud SDK Installation
# -------------------------------------------
echo "[5/7] Installing Google Cloud SDK..."
if ! command -v gcloud &> /dev/null; then
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    sudo apt-get update && sudo apt-get install -y google-cloud-cli
else
    echo "Google Cloud SDK already installed."
fi

# -------------------------------------------
# C++ Build Tools (for pybind11)
# -------------------------------------------
echo "[6/7] Installing C++ build tools..."
sudo apt-get install -y build-essential cmake g++

# -------------------------------------------
# Project Setup
# -------------------------------------------
echo "[7/7] Setting up project..."
APP_DIR="/app/ofc-training"
sudo mkdir -p $APP_DIR
sudo chown -R $USER:$USER /app

# 手動でファイルをコピーする場合:
echo "Please copy project files to $APP_DIR"
echo ""
echo "例: gcloud compute scp --recurse ./OFC\\ NN/* INSTANCE_NAME:$APP_DIR/"

# -------------------------------------------
# Environment Setup
# -------------------------------------------
if [ ! -f "$APP_DIR/.env" ]; then
    cat > "$APP_DIR/.env.example" << 'EOF'
# GCP Configuration
GCS_BUCKET=your-bucket-name
GCP_PROJECT=your-project-id
GCP_REGION=asia-northeast1

# Discord Notification
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxxx

# Training Configuration
TOTAL_TIMESTEPS=20000000
CHECKPOINT_FREQ=200000
NOTIFICATION_FREQ=100000
EOF
    echo "Created .env.example. Please copy to .env and edit."
fi

# -------------------------------------------
# Python Environment Setup
# -------------------------------------------
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "$APP_DIR/.venv" ]; then
    python3.9 -m venv $APP_DIR/.venv
fi

# -------------------------------------------
# Helper Scripts
# -------------------------------------------
cat > /usr/local/bin/ofc-start << 'EOF'
#!/bin/bash
cd /app/ofc-training
source .venv/bin/activate
nohup python src/python/train_gcp_phase7.py > training.log 2>&1 &
echo $! > training.pid
echo "Training started with PID: $(cat training.pid)"
echo "Logs: tail -f training.log"
EOF
sudo chmod +x /usr/local/bin/ofc-start

cat > /usr/local/bin/ofc-stop << 'EOF'
#!/bin/bash
cd /app/ofc-training
if [ -f training.pid ]; then
    kill $(cat training.pid) 2>/dev/null || true
    rm training.pid
    echo "Training stopped."
else
    echo "No training process found."
fi
EOF
sudo chmod +x /usr/local/bin/ofc-stop

cat > /usr/local/bin/ofc-status << 'EOF'
#!/bin/bash
cd /app/ofc-training
if [ -f training.pid ] && ps -p $(cat training.pid) > /dev/null 2>&1; then
    echo "Training is running (PID: $(cat training.pid))"
    echo ""
    echo "=== Recent logs ==="
    tail -30 training.log
else
    echo "Training is not running."
    if [ -f training.log ]; then
        echo ""
        echo "=== Last logs ==="
        tail -30 training.log
    fi
fi
EOF
sudo chmod +x /usr/local/bin/ofc-status

cat > /usr/local/bin/ofc-logs << 'EOF'
#!/bin/bash
cd /app/ofc-training
tail -f training.log
EOF
sudo chmod +x /usr/local/bin/ofc-logs

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
echo "     gcloud compute scp --recurse ./OFC\\ NN/* INSTANCE_NAME:$APP_DIR/"
echo ""
echo "  2. Setup Python environment:"
echo "     cd $APP_DIR"
echo "     source .venv/bin/activate"
echo "     pip install -r requirements.txt"
echo "     python setup.py build_ext --inplace"
echo ""
echo "  3. Configure GCS:"
echo "     gcloud auth application-default login"
echo "     gsutil mb gs://YOUR_BUCKET_NAME"
echo ""
echo "  4. Edit .env file:"
echo "     cp .env.example .env"
echo "     vim .env"
echo ""
echo "  5. Start training:"
echo "     ofc-start"
echo ""
echo "Available commands:"
echo "  ofc-start  - Start training"
echo "  ofc-stop   - Stop training"
echo "  ofc-status - Check status and logs"
echo "  ofc-logs   - Follow training logs"
echo ""
