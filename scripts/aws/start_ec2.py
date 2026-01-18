import boto3
import os
from dotenv import load_dotenv

load_dotenv()

ec2 = boto3.client('ec2', region_name=os.getenv('AWS_REGION', 'ap-northeast-1'))

def find_and_start_instance():
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Name', 'Values': ['ofc-training']}
        ]
    )
    
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instances.append(instance)
            
    if not instances:
        print("No instance found with Name=ofc-training")
        return
    
    instance = instances[0]
    instance_id = instance['InstanceId']
    state = instance['State']['Name']
    
    print(f"Found instance: {instance_id} (State: {state})")
    
    if state == 'stopped':
        print(f"Starting instance {instance_id}...")
        ec2.start_instances(InstanceIds=[instance_id])
        print("Start signal sent.")
    elif state == 'running':
        print("Instance is already running.")
    else:
        print(f"Instance is in state: {state}")

if __name__ == "__main__":
    find_and_start_instance()
