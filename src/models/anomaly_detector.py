import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Add src directory to Python's import path.
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.validator import load_dataset
from features.engineering import (
    engineer_features,
    get_ml_features,
    validate_ml_features,
)


DEFAULT_CONTAMINATION = 0.03
DEFAULT_RANDOM_STATE = 42


class AnomalyDetector:
    """
    Isolation Forest based anomaly detector.

    The model operates only on engineered ML features.
    Ground-truth anomaly labels are never passed to the model.
    """

    def __init__(
        self,
        contamination: float = DEFAULT_CONTAMINATION,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        if not 0 < contamination <= 0.5:
            raise ValueError(
                "Contamination must be between "
                "0 and 0.5."
            )

        self.contamination = contamination
        self.random_state = random_state

        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )

        self.feature_columns: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "AnomalyDetector":
        """
        Fit the Isolation Forest model.
        """

        self.feature_columns = get_ml_features(df)

        validate_ml_features(
            df,
            self.feature_columns,
        )

        X = df[self.feature_columns]

        self.model.fit(X)

        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate anomaly predictions and scores.
        """

        if not self.feature_columns:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        validate_ml_features(
            df,
            self.feature_columns,
        )

        X = df[self.feature_columns]

        sklearn_predictions = (
            self.model.predict(X)
        )

        anomaly_scores = (
            self.model.decision_function(X)
        )

        result = df.copy()

        result["anomaly_score"] = (
            anomaly_scores
        )

        result["is_predicted_anomaly"] = (
            sklearn_predictions == -1
        ).astype(int)

        result["model_prediction"] = (
            sklearn_predictions
        )

        return result

    def fit_predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit the model and generate predictions.
        """

        self.fit(df)

        return self.predict(df)


def evaluate_predictions(
    df: pd.DataFrame,
) -> dict:
    """
    Evaluate model predictions against the
    synthetic dataset's ground-truth labels.

    Ground truth is used ONLY for evaluation.
    """

    if "is_injected_anomaly" not in df.columns:
        raise ValueError(
            "Ground-truth anomaly column is missing."
        )

    if "is_predicted_anomaly" not in df.columns:
        raise ValueError(
            "Model prediction column is missing."
        )

    y_true = (
        df["is_injected_anomaly"]
        .astype(int)
    )

    y_pred = (
        df["is_predicted_anomaly"]
        .astype(int)
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix,
        "classification_report": (
            classification_report(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


def print_model_report(
    predictions: pd.DataFrame,
    evaluation: dict,
) -> None:
    """
    Print anomaly detection results.
    """

    total_records = len(predictions)

    predicted_anomalies = int(
        predictions[
            "is_predicted_anomaly"
        ].sum()
    )

    actual_anomalies = int(
        predictions[
            "is_injected_anomaly"
        ].sum()
    )

    predicted_rate = (
        predicted_anomalies
        / total_records
        * 100
    )

    actual_rate = (
        actual_anomalies
        / total_records
        * 100
    )

    print()
    print("=" * 80)
    print("ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 80)

    print(
        f"Total records:          {total_records:,}"
    )

    print(
        f"Ground-truth anomalies: {actual_anomalies:,}"
    )

    print(
        f"Ground-truth rate:      {actual_rate:.2f}%"
    )

    print(
        f"Predicted anomalies:    {predicted_anomalies:,}"
    )

    print(
        f"Predicted rate:         {predicted_rate:.2f}%"
    )

    print()
    print("MODEL CONFIGURATION")
    print("-" * 80)

    print(
        "Algorithm:              Isolation Forest"
    )

    print(
        "Estimators:             200"
    )

    print(
        "Contamination:          "
        f"{DEFAULT_CONTAMINATION:.2%}"
    )

    print(
        "Random state:           "
        f"{DEFAULT_RANDOM_STATE}"
    )

    print()
    print("EVALUATION")
    print("-" * 80)

    print(
        f"Precision:              "
        f"{evaluation['precision']:.4f}"
    )

    print(
        f"Recall:                 "
        f"{evaluation['recall']:.4f}"
    )

    print(
        f"F1 Score:               "
        f"{evaluation['f1_score']:.4f}"
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 80)

    print(
        evaluation[
            "confusion_matrix"
        ]
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 80)

    print(
        evaluation[
            "classification_report"
        ]
    )

    print("=" * 80)


def main():
    project_root = (
        Path(__file__).resolve().parents[2]
    )

    dataset_path = (
        project_root
        / "data"
        / "synthetic_retail_data.csv"
    )

    # Load raw dataset.
    df = load_dataset(
        dataset_path
    )

    # Engineer features.
    engineered_df = engineer_features(
        df
    )

    # Fit and predict.
    detector = AnomalyDetector()

    predictions = detector.fit_predict(
        engineered_df
    )

    # Evaluate ONLY after prediction.
    evaluation = evaluate_predictions(
        predictions
    )

    print_model_report(
        predictions,
        evaluation,
    )


if __name__ == "__main__":
    main()