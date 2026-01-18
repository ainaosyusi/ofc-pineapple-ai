import boto3
import os
from dotenv import load_dotenv

def debug_aws():
    print("--- AWS Debug Info ---")
    load_dotenv()
    
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'ap-northeast-1')
    bucket_name = os.getenv('S3_BUCKET')
    
    print(f"Access Key ID (Masked): {access_key[:5] if access_key else 'None'}...")
    print(f"Region: {region}")
    print(f"Target Bucket: {bucket_name}")
    
    try:
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        ec2 = session.client('ec2')
        s3 = session.client('s3')
        
        print("\n[EC2 Instances]")
        instances = ec2.describe_instances()
        for res in instances['Reservations']:
            for inst in res['Instances']:
                name = next((t['Value'] for t in inst.get('Tags', []) if t['Key'] == 'Name'), 'N/A')
                print(f"- {inst['InstanceId']} ({name}): {inst['State']['Name']} | IP: {inst.get('PublicIpAddress', 'N/A')}")
        
        print("\n[S3 Buckets]")
        buckets = s3.list_buckets()['Buckets']
        for b in buckets:
            print(f"- {b['Name']}")
            if b['Name'] == bucket_name:
                print(f"  (Matched target bucket!)")
        
    except Exception as e:
        print(f"\n[Error] {str(e)}")

if __name__ == "__main__":
    debug_aws()
