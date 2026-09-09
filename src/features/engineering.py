import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Add the src directory to Python's import path.
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from analysis.validator import load_dataset


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-oriented numerical features
    for downstream analytics and anomaly detection.

    Ground-truth anomaly labels are intentionally
    excluded from feature engineering.
    """

    features = df.copy()

    # ---------------------------------------------------------
    # Revenue and transaction features
    # ---------------------------------------------------------

    features["transaction_value"] = (
        features["quantity"]
        * features["unit_price"]
    )

    features["discount_amount"] = (
        features["transaction_value"]
        * features["discount_pct"]
        / 100
    )

    features["discount_ratio"] = (
        features["discount_amount"]
        / features["transaction_value"].replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # Profit and cost features
    # ---------------------------------------------------------

    features["profit_margin"] = (
        features["profit"]
        / features["revenue"].replace(0, np.nan)
    )

    features["cost_to_revenue_ratio"] = (
        (
            features["shipping_cost"]
            + features["operational_cost"]
        )
        / features["revenue"].replace(0, np.nan)
    )

    features["shipping_to_revenue_ratio"] = (
        features["shipping_cost"]
        / features["revenue"].replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # Delivery and satisfaction features
    # ---------------------------------------------------------

    features["delivery_delay_score"] = (
        features["delivery_days"]
        / features["delivery_days"].median()
    )

    features["satisfaction_risk"] = (
        5 - features["satisfaction_score"]
    )

    # ---------------------------------------------------------
    # Return features
    # ---------------------------------------------------------

    features["return_ratio"] = (
        features["return_amount"]
        / features["revenue"].replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # Quantity behavior
    # ---------------------------------------------------------

    features["quantity_per_order"] = (
        features["quantity"]
    )

    # ---------------------------------------------------------
    # Date-based features
    # ---------------------------------------------------------

    order_dates = pd.to_datetime(
        features["order_date"],
        errors="coerce",
    )

    features["order_month"] = (
        order_dates.dt.month
    )

    features["order_day_of_week"] = (
        order_dates.dt.dayofweek
    )

    # ---------------------------------------------------------
    # Clean numerical values
    # ---------------------------------------------------------

    engineered_columns = [
        "transaction_value",
        "discount_amount",
        "discount_ratio",
        "profit_margin",
        "cost_to_revenue_ratio",
        "shipping_to_revenue_ratio",
        "delivery_delay_score",
        "satisfaction_risk",
        "return_ratio",
        "quantity_per_order",
        "order_month",
        "order_day_of_week",
    ]

    features[engineered_columns] = (
        features[engineered_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return features


def get_ml_features(
    engineered_df: pd.DataFrame,
) -> list[str]:
    """
    Return the numerical features that will be used
    by the anomaly detection model.

    Ground-truth labels are deliberately excluded.
    """

    ml_features = [
        "quantity",
        "unit_price",
        "discount_pct",
        "revenue",
        "return_amount",
        "delivery_days",
        "satisfaction_score",
        "shipping_cost",
        "operational_cost",
        "profit",
        "transaction_value",
        "discount_amount",
        "discount_ratio",
        "profit_margin",
        "cost_to_revenue_ratio",
        "shipping_to_revenue_ratio",
        "delivery_delay_score",
        "satisfaction_risk",
        "return_ratio",
        "quantity_per_order",
        "order_month",
        "order_day_of_week",
    ]

    available_features = [
        column
        for column in ml_features
        if column in engineered_df.columns
    ]

    return available_features


def validate_ml_features(
    engineered_df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Validate the feature matrix before sending it
    to an ML model.
    """

    if not feature_columns:
        raise ValueError(
            "No ML features are available."
        )

    missing_columns = [
        column
        for column in feature_columns
        if column not in engineered_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing ML features: "
            + ", ".join(missing_columns)
        )

    feature_matrix = engineered_df[
        feature_columns
    ]

    if feature_matrix.isnull().any().any():
        raise ValueError(
            "ML feature matrix contains missing values."
        )

    if not np.isfinite(
        feature_matrix.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "ML feature matrix contains "
            "non-finite values."
        )


def print_feature_report(
    original_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Print a summary of the feature engineering stage.
    """

    print()
    print("=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    print(
        f"Original columns:   {len(original_df.columns)}"
    )

    print(
        f"Engineered columns: {len(engineered_df.columns)}"
    )

    print(
        f"New columns:        "
        f"{len(engineered_df.columns) - len(original_df.columns)}"
    )

    print(
        f"ML features:        {len(feature_columns)}"
    )

    print()
    print("ML FEATURES")
    print("-" * 80)

    for index, feature in enumerate(
        feature_columns,
        start=1,
    ):
        print(
            f"{index:>2}. {feature}"
        )

    print()
    print("ENGINEERED FEATURE PREVIEW")
    print("-" * 80)

    preview_columns = [
        "order_id",
        "transaction_value",
        "discount_amount",
        "discount_ratio",
        "profit_margin",
        "cost_to_revenue_ratio",
        "shipping_to_revenue_ratio",
        "delivery_delay_score",
        "satisfaction_risk",
        "return_ratio",
    ]

    available_preview = [
        column
        for column in preview_columns
        if column in engineered_df.columns
    ]

    print(
        engineered_df[
            available_preview
        ].head(5).to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "STATUS: PASS"
    )
    print(
        "Feature engineering completed successfully."
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

    df = load_dataset(
        dataset_path
    )

    engineered_df = engineer_features(
        df
    )

    feature_columns = get_ml_features(
        engineered_df
    )

    validate_ml_features(
        engineered_df,
        feature_columns,
    )

    print_feature_report(
        df,
        engineered_df,
        feature_columns,
    )


if __name__ == "__main__":
    main()