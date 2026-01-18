import boto3
import os
from dotenv import load_dotenv

load_dotenv()

ec2 = boto3.client('ec2', region_name=os.getenv('AWS_REGION', 'ap-northeast-1'))

def check_status():
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Name', 'Values': ['ofc-training']}
        ]
    )
    
    if not response['Reservations']:
        print("No reservations found.")
        return
    
    found = False
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            found = True
            instance_id = instance['InstanceId']
            state = instance['State']['Name']
            public_ip = instance.get('PublicIpAddress', 'N/A')
            launch_time = instance['LaunchTime']
            print(f"ID: {instance_id}")
            print(f"State: {state}")
            print(f"Public IP: {public_ip}")
            print(f"Launch Time: {launch_time}")
    
    if not found:
        print("No instances found matching the filter.")

if __name__ == "__main__":
    check_status()
