# Fraud Detection with Amazon SageMaker

An end-to-end fraud detection pipeline built on Amazon SageMaker, using the
[Credit Card Fraud Detection dataset (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

This project is a learning-focused walkthrough of core SageMaker capabilities:
data processing, training with a built-in algorithm, hyperparameter tuning,
model deployment, batch inference, and pipeline orchestration.

## Problem

Classify credit card transactions as fraudulent (`Class = 1`) or legitimate
(`Class = 0`). The dataset is highly imbalanced — frauds account for only
~0.17% of transactions — so the project also covers imbalance-aware
evaluation (Precision-Recall AUC rather than plain accuracy).

## Architecture

```
Raw CSV (S3) --> SageMaker Processing (split + scale)
              --> SageMaker Training (built-in XGBoost)
              --> SageMaker HPO (tune hyperparameters)
              --> Evaluation (Processing job, PR-AUC)
              --> Model Registry (register best model)
              --> Real-time Endpoint (single-transaction scoring)
              --> Batch Transform (bulk scoring)
```

All of the above is wired together as a **SageMaker Pipeline** in
`pipeline/pipeline.py`, so the whole thing can be triggered with a single
`pipeline.start()` call.

## Project structure

```
fraud-detection-sagemaker/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                     # local scratch space (raw CSV goes here, gitignored)
├── processing/
│   ├── preprocessing.py      # SageMaker Processing script: clean/split/scale
│   └── evaluation.py         # SageMaker Processing script: compute PR-AUC on test set
├── pipeline/
│   └── pipeline.py           # Defines and builds the SageMaker Pipeline
├── notebooks/
│   └── 01_run_pipeline.ipynb # Entry point — run this from SageMaker Studio
└── src/
    └── config.py             # Shared constants (bucket, role, instance types, etc.)
```

## Setup

1. **Get the data**
   Download `creditcard.csv` from Kaggle:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
   ```
   (Requires a Kaggle account and API token — see
   https://www.kaggle.com/docs/api)

2. **Configure AWS**
   - Have an AWS account with a SageMaker execution role (one with S3,
     ECR, and CloudWatch access — SageMaker Studio can generate this for you).
   - Edit `src/config.py` with your S3 bucket name and IAM role ARN.

3. **Upload raw data to S3**
   ```bash
   aws s3 cp data/creditcard.csv s3://<your-bucket>/fraud-detection/raw/creditcard.csv
   ```

4. **Run the pipeline**
   Open `notebooks/01_run_pipeline.ipynb` in SageMaker Studio and run all
   cells. This builds and executes the full pipeline: processing → training
   → tuning → evaluation → conditional registration.

5. **Deploy and test**
   The notebook includes cells to deploy the registered model to a
   real-time endpoint and send a sample transaction for scoring, plus a
   batch transform example.

## Cost notes

- Stop/delete any SageMaker Studio apps, endpoints, and notebook instances
  when you're done — SageMaker resources bill by the second while running.
- Set a billing alarm in AWS Budgets before running this project.

## Status

- [ ] Data uploaded to S3
- [ ] Processing step tested
- [ ] Training step tested
- [ ] HPO tuning job run
- [ ] Model registered
- [ ] Endpoint deployed
- [ ] Batch transform tested
