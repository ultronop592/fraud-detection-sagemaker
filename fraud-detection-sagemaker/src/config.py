"""
Shared configuration for the fraud detection SageMaker project.
Edit BUCKET and ROLE before running anything.
"""

import boto3
import sagemaker

# ---- Required: edit these ----
BUCKET = "your-bucket-name"          # e.g. "my-sagemaker-fraud-project"
ROLE = "arn:aws:iam::<account-id>:role/service-role/AmazonSageMaker-ExecutionRole-XXXX"

# ---- S3 layout ----
PREFIX = "fraud-detection"
RAW_DATA_S3 = f"s3://{BUCKET}/{PREFIX}/raw/creditcard.csv"
PROCESSED_TRAIN_S3 = f"s3://{BUCKET}/{PREFIX}/processed/train"
PROCESSED_VALIDATION_S3 = f"s3://{BUCKET}/{PREFIX}/processed/validation"
PROCESSED_TEST_S3 = f"s3://{BUCKET}/{PREFIX}/processed/test"
MODEL_OUTPUT_S3 = f"s3://{BUCKET}/{PREFIX}/model-output"
EVALUATION_OUTPUT_S3 = f"s3://{BUCKET}/{PREFIX}/evaluation"
BATCH_INPUT_S3 = f"s3://{BUCKET}/{PREFIX}/batch/input"
BATCH_OUTPUT_S3 = f"s3://{BUCKET}/{PREFIX}/batch/output"

# ---- Instance types (small/cheap by default) ----
PROCESSING_INSTANCE_TYPE = "ml.m5.large"
TRAINING_INSTANCE_TYPE = "ml.m5.large"
INFERENCE_INSTANCE_TYPE = "ml.m5.large"
BATCH_TRANSFORM_INSTANCE_TYPE = "ml.m5.large"

# ---- Model package group (for the Model Registry) ----
MODEL_PACKAGE_GROUP_NAME = "FraudDetectionModelPackageGroup"
PIPELINE_NAME = "FraudDetectionPipeline"

# ---- Helpers ----
def get_session():
    boto_session = boto3.Session()
    return sagemaker.Session(boto_session=boto_session, default_bucket=BUCKET)
