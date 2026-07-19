"""
SageMaker Processing script that evaluates a trained XGBoost model
against the held-out test set and writes a metrics report.

Because the dataset is highly imbalanced, we report Precision-Recall AUC
(average precision) alongside accuracy and ROC-AUC -- accuracy alone is
misleading when frauds are ~0.17% of all transactions.

Expects:
  /opt/ml/processing/model/model.tar.gz   (trained model artifact)
  /opt/ml/processing/test/test.csv        (label in first column, no header)

Writes:
  /opt/ml/processing/evaluation/evaluation.json
"""

import json
import os
import tarfile

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)


def main():
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="/opt/ml/processing/model")

    model = xgb.Booster()
    model.load_model("/opt/ml/processing/model/xgboost-model")

    test_df = pd.read_csv("/opt/ml/processing/test/test.csv", header=None)
    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    report = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy_score(y_test, y_pred)},
            "roc_auc": {"value": roc_auc_score(y_test, y_pred_proba)},
            "pr_auc": {"value": average_precision_score(y_test, y_pred_proba)},
        }
    }

    print(json.dumps(report, indent=2))

    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/evaluation.json", "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
