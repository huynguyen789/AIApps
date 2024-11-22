import boto3
import streamlit as st

# Load AWS credentials from streamlit secrets
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_DEFAULT_REGION"]

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

# List buckets
response = s3_client.list_buckets()
buckets = response['Buckets']

# Print bucket names
print("Your S3 buckets:")
for bucket in buckets:
    print(f"- {bucket['Name']}")

# # Create new bucket
# bucket_name = 'ai-memory-huy'
# try:
#     if aws_region == 'us-east-1':
#         s3_client.create_bucket(Bucket=bucket_name)
#     else:
#         s3_client.create_bucket(
#             Bucket=bucket_name,
#             CreateBucketConfiguration={'LocationConstraint': aws_region}
#         )
#     print(f"Successfully created bucket: {bucket_name}")
# except s3_client.exceptions.BucketAlreadyExists:
#     print(f"Bucket {bucket_name} already exists")
# except s3_client.exceptions.BucketAlreadyOwnedByYou:
#     print(f"Bucket {bucket_name} already owned by you")
# except Exception as e:
#     print(f"Error creating bucket: {str(e)}")
