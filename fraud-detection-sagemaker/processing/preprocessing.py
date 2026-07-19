"""
SageMaker Processing script for the credit card fraud dataset.

Reads the raw CSV (mounted at /opt/ml/processing/input), splits it into
train/validation/test, scales the 'Amount' feature, and writes CSV files
formatted for SageMaker's built-in XGBoost algorithm:
  - no header, no index
  - label ('Class') as the first column

Output channels:
  /opt/ml/processing/output/train/train.csv
  /opt/ml/processing/output/validation/validation.csv
  /opt/ml/processing/output/test/test.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = "/opt/ml/processing/input/creditcard.csv"
    output_dir = "/opt/ml/processing/output"

    df = pd.read_csv(input_path)

    # Scale 'Amount' (Time is dropped -- it's just seconds since first
    # transaction and carries little standalone signal for this dataset;
    # V1-V28 are already PCA components and don't need scaling)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df = df.drop(columns=["Time"])

    # Put label first, as required by SageMaker's built-in XGBoost
    label_col = "Class"
    feature_cols = [c for c in df.columns if c != label_col]
    df = df[[label_col] + feature_cols]

    # Stratified split so the (rare) fraud class is represented in every split
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_size,
        stratify=df[label_col],
        random_state=args.random_state,
    )
    relative_val_size = args.validation_size / (1 - args.train_size)
    validation_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df[label_col],
        random_state=args.random_state,
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train fraud rate: {train_df[label_col].mean():.5f}")
    print(f"Validation fraud rate: {validation_df[label_col].mean():.5f}")
    print(f"Test fraud rate: {test_df[label_col].mean():.5f}")

    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/validation", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    train_df.to_csv(f"{output_dir}/train/train.csv", header=False, index=False)
    validation_df.to_csv(
        f"{output_dir}/validation/validation.csv", header=False, index=False
    )
    test_df.to_csv(f"{output_dir}/test/test.csv", header=False, index=False)


if __name__ == "__main__":
    main()
