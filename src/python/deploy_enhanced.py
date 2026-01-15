import boto3
import os
import time
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
REGION = os.getenv('AWS_REGION', 'ap-northeast-1')
INSTANCE_TYPE = 'm7i-flex.large'
KEY_NAME = 'ofc-training-key'
TAG_NAME = 'ofc-training-enhanced'
AMI_ID = 'ami-0aec5ae807cea9ce0'  # Replicating the running instance's AMI
SUBNET_ID = 'subnet-093bdafd1b12512e4'
SECURITY_GROUP_IDS = ['sg-0d7c01aa0c9a54487']

ec2 = boto3.client('ec2', region_name=REGION)

def launch_enhanced_instance():
    print(f"🚀 Replicating instance type: {INSTANCE_TYPE}...")
    
    # User data to install docker and prepare environment (just in case AMI is clean)
    user_data = """#!/bin/bash
apt-get update
apt-get install -y docker.io docker-compose rsync
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu
mkdir -p /home/ubuntu/OFC-NN
chown ubuntu:ubuntu /home/ubuntu/OFC-NN
"""

    run_instances = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        MinCount=1,
        MaxCount=1,
        SubnetId=SUBNET_ID,
        SecurityGroupIds=SECURITY_GROUP_IDS,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [{'Key': 'Name', 'Value': TAG_NAME}]
            }
        ],
        UserData=user_data
    )

    instance_id = run_instances['Instances'][0]['InstanceId']
    print(f"✅ Instance launched: {instance_id}")
    
    # Wait for instance to be running to get IP
    print("Waiting for instance to be running...")
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    
    description = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = description['Reservations'][0]['Instances'][0].get('PublicIpAddress')
    
    print(f"\n✨ Instance is ready!")
    print(f"Instance ID: {instance_id}")
    print(f"Public IP:   {public_ip}")
    print(f"\nNext steps:")
    print(f"1. Wait a few minutes for docker installation to complete.")
    print(f"2. Run rsync to transfer code:")
    print(f"   rsync -avz -e 'ssh -i {KEY_NAME}.pem' --exclude 'models/*.zip' --exclude '__pycache__' ./ ubuntu@{public_ip}:/home/ubuntu/OFC-NN/")
    print(f"3. SSH and start training:")
    print(f"   ssh -i {KEY_NAME}.pem ubuntu@{public_ip}")
    print(f"   cd /home/ubuntu/OFC-NN")
    print(f"   docker-compose up -d enhanced-phase3")

if __name__ == "__main__":
    launch_enhanced_instance()
