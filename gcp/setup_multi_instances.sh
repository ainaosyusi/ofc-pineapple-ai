#!/bin/bash
# GCP Multi-Instance Setup for OFC AI Variants
# 3つの異なるAIバリアントを並行学習

set -e

PROJECT_ID="ofcnntraining"
ZONE="asia-northeast1-b"
MACHINE_TYPE="n2-standard-4"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="50GB"

# 既存インスタンス (Phase 8 Self-Play)
INSTANCE_1="ofc-training"

# 新規インスタンス
INSTANCE_2="ofc-aggressive"
INSTANCE_3="ofc-teacher"

echo "============================================"
echo "OFC AI Multi-Instance Setup"
echo "============================================"
echo ""
echo "Creating instances:"
echo "  1. $INSTANCE_1 (既存) - Phase 8 Self-Play"
echo "  2. $INSTANCE_2 (新規) - Aggressive Variant"
echo "  3. $INSTANCE_3 (新規) - Teacher Learning"
echo ""
echo "Estimated cost: ~\$0.57/hour total"
echo ""

# Instance 2: Aggressive
echo "Creating $INSTANCE_2..."
gcloud compute instances create $INSTANCE_2 \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv git
' \
    2>/dev/null || echo "$INSTANCE_2 may already exist"

# Instance 3: Teacher
echo "Creating $INSTANCE_3..."
gcloud compute instances create $INSTANCE_3 \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv git
' \
    2>/dev/null || echo "$INSTANCE_3 may already exist"

echo ""
echo "============================================"
echo "Instances created. Getting IP addresses..."
echo "============================================"

sleep 5

# IPアドレス取得
IP_1=$(gcloud compute instances describe $INSTANCE_1 --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "N/A")
IP_2=$(gcloud compute instances describe $INSTANCE_2 --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "N/A")
IP_3=$(gcloud compute instances describe $INSTANCE_3 --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "N/A")

echo ""
echo "Instance IPs:"
echo "  $INSTANCE_1: $IP_1"
echo "  $INSTANCE_2: $IP_2"
echo "  $INSTANCE_3: $IP_3"
echo ""
echo "Next steps:"
echo "  1. Run setup_instance.sh on each new instance"
echo "  2. Upload training code"
echo "  3. Start training"
echo ""
echo "Quick commands:"
echo "  ssh naoai@$IP_2 'cd ~/ofc-training && ...'  # Aggressive"
echo "  ssh naoai@$IP_3 'cd ~/ofc-training && ...'  # Teacher"
