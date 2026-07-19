"""
Defines the SageMaker Pipeline for the fraud detection project:

  Processing (split/scale)
    -> Hyperparameter Tuning (built-in XGBoost)
    -> Processing (evaluation, PR-AUC)
    -> Condition (PR-AUC above threshold?)
        -> Register model in Model Registry

Run get_pipeline() and call .upsert() + .start() from the notebook.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.tuning_step import TuningStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.model import Model
from sagemaker.image_uris import retrieve

from src import config


def get_pipeline(session=None, role=None):
    sagemaker_session = session or config.get_session()
    role = role or config.ROLE

    # ---- Step 1: Processing (split + scale) ----
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=config.PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        base_job_name="fraud-preprocessing",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    step_process = ProcessingStep(
        name="PreprocessFraudData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=config.RAW_DATA_S3,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test"),
        ],
        code="processing/preprocessing.py",
    )

    # ---- Step 2: Hyperparameter Tuning (built-in XGBoost) ----
    xgboost_image = retrieve("xgboost", sagemaker_session.boto_region_name, version="1.7-1")

    xgb_estimator = Estimator(
        image_uri=xgboost_image,
        instance_type=config.TRAINING_INSTANCE_TYPE,
        instance_count=1,
        output_path=config.MODEL_OUTPUT_S3,
        role=role,
        sagemaker_session=sagemaker_session,
    )
    xgb_estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="aucpr",   # PR-AUC, appropriate for imbalanced data
        num_round=200,
        # Roughly (# negative / # positive) -- helps XGBoost weight the rare fraud class
        scale_pos_weight=577,
    )

    hyperparameter_ranges = {
        "eta": ContinuousParameter(0.01, 0.3),
        "max_depth": IntegerParameter(3, 10),
        "min_child_weight": ContinuousParameter(1, 10),
        "subsample": ContinuousParameter(0.5, 1.0),
    }

    tuner = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name="validation:aucpr",
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type="Maximize",
        max_jobs=10,
        max_parallel_jobs=2,
    )

    step_tune = TuningStep(
        name="TuneFraudXGBoost",
        tuner=tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ---- Step 3: Evaluation ----
    script_processor = ScriptProcessor(
        image_uri=xgboost_image,
        command=["python3"],
        instance_type=config.PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        base_job_name="fraud-evaluation",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_evaluate = ProcessingStep(
        name="EvaluateFraudModel",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=step_tune.get_top_model_s3_uri(top_k=0, s3_bucket=config.BUCKET),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=config.EVALUATION_OUTPUT_S3,
            )
        ],
        code="processing/evaluation.py",
        property_files=[evaluation_report],
    )

    # ---- Step 4: Register model, gated on PR-AUC threshold ----
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{config.EVALUATION_OUTPUT_S3}/evaluation.json",
            content_type="application/json",
        )
    )

    model = Model(
        image_uri=xgboost_image,
        model_data=step_tune.get_top_model_s3_uri(top_k=0, s3_bucket=config.BUCKET),
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_register = ModelStep(
        name="RegisterFraudModel",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=[config.INFERENCE_INSTANCE_TYPE],
            transform_instances=[config.BATCH_TRANSFORM_INSTANCE_TYPE],
            model_package_group_name=config.MODEL_PACKAGE_GROUP_NAME,
            approval_status="PendingManualApproval",
            model_metrics=model_metrics,
        ),
    )

    step_condition = ConditionStep(
        name="CheckPRAUCThreshold",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="binary_classification_metrics.pr_auc.value",
                ),
                right=0.5,  # tune this threshold as you see fit
            )
        ],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=config.PIPELINE_NAME,
        steps=[step_process, step_tune, step_evaluate, step_condition],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    print(pipeline.definition())
