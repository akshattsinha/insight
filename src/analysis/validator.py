from pathlib import Path

import pandas as pd


# =============================================================================
# REQUIRED BUSINESS COLUMNS
# =============================================================================

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "region",
    "channel",
    "customer_segment",
    "category",
    "product",
    "quantity",
    "unit_price",
    "discount_pct",
    "revenue",
    "return_status",
    "return_amount",
    "delivery_days",
    "satisfaction_score",
    "shipping_cost",
    "operational_cost",
    "profit",
    "payment_method",
]


# =============================================================================
# OPTIONAL SYNTHETIC GROUND-TRUTH COLUMNS
# =============================================================================

GROUND_TRUTH_COLUMNS = [
    "is_injected_anomaly",
    "anomaly_type",
]


# =============================================================================
# EXPECTED NUMERIC COLUMNS
# =============================================================================

EXPECTED_NUMERIC_COLUMNS = [
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
]


# =============================================================================
# EXPECTED CATEGORICAL COLUMNS
# =============================================================================

EXPECTED_CATEGORICAL_COLUMNS = [
    "region",
    "channel",
    "customer_segment",
    "category",
    "product",
    "return_status",
    "payment_method",
]


# =============================================================================
# VALID BUSINESS VALUES
# =============================================================================

VALID_REGIONS = {
    "North",
    "South",
    "East",
    "West",
    "Central",
}


VALID_CHANNELS = {
    "Website",
    "Mobile App",
    "Marketplace",
}


VALID_CUSTOMER_SEGMENTS = {
    "New",
    "Regular",
    "Premium",
}


VALID_RETURN_STATUS = {
    "Returned",
    "Not Returned",
}


VALID_ANOMALY_TYPES = {
    "none",
    "revenue_spike",
    "unusual_discount",
    "delivery_delay",
    "high_return",
    "profit_loss",
}


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load a CSV dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema(
    df: pd.DataFrame,
) -> list[str]:
    """
    Validate required business columns.

    Synthetic ground-truth columns are optional.
    """

    errors = []

    actual_columns = set(df.columns)
    required_columns = set(REQUIRED_COLUMNS)

    missing_columns = (
        required_columns - actual_columns
    )

    if missing_columns:
        errors.append(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return errors


# =============================================================================
# GROUND-TRUTH DETECTION
# =============================================================================

def has_ground_truth(
    df: pd.DataFrame,
) -> bool:
    """
    Return True when the dataset contains the
    synthetic anomaly ground-truth columns.
    """

    return all(
        column in df.columns
        for column in GROUND_TRUTH_COLUMNS
    )


# =============================================================================
# MISSING VALUE VALIDATION
# =============================================================================

def validate_missing_values(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check for missing values.
    """

    errors = []

    missing_counts = df.isna().sum()

    columns_with_missing = (
        missing_counts[
            missing_counts > 0
        ]
    )

    for column, count in (
        columns_with_missing.items()
    ):
        errors.append(
            f"{column}: {count} missing values"
        )

    return errors


# =============================================================================
# DUPLICATE VALIDATION
# =============================================================================

def validate_duplicates(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check order IDs for duplicates.
    """

    errors = []

    if "order_id" not in df.columns:
        return errors

    duplicate_count = (
        df["order_id"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        errors.append(
            f"Duplicate order IDs: "
            f"{duplicate_count}"
        )

    return errors


# =============================================================================
# NUMERIC RANGE VALIDATION
# =============================================================================

def validate_numeric_ranges(
    df: pd.DataFrame,
) -> list[str]:
    """
    Validate important business metric ranges.
    """

    errors = []

    range_rules = {
        "quantity": (1, None),
        "unit_price": (0, None),
        "discount_pct": (0, 100),
        "revenue": (0, None),
        "return_amount": (0, None),
        "delivery_days": (1, None),
        "satisfaction_score": (1, 5),
        "shipping_cost": (0, None),
        "operational_cost": (0, None),
    }

    for column, (
        minimum,
        maximum,
    ) in range_rules.items():

        if column not in df.columns:
            continue

        if minimum is not None:

            invalid = (
                df[column] < minimum
            )

            if invalid.any():

                errors.append(
                    f"{column}: "
                    f"{invalid.sum()} values below "
                    f"minimum {minimum}"
                )

        if maximum is not None:

            invalid = (
                df[column] > maximum
            )

            if invalid.any():

                errors.append(
                    f"{column}: "
                    f"{invalid.sum()} values above "
                    f"maximum {maximum}"
                )

    return errors


# =============================================================================
# CATEGORY VALIDATION
# =============================================================================

def validate_categories(
    df: pd.DataFrame,
) -> list[str]:
    """
    Validate known business categorical values.

    Synthetic anomaly_type is validated only when
    ground truth is available.
    """

    errors = []

    category_rules = {
        "region": VALID_REGIONS,
        "channel": VALID_CHANNELS,
        "customer_segment": VALID_CUSTOMER_SEGMENTS,
        "return_status": VALID_RETURN_STATUS,
        "anomaly_type": VALID_ANOMALY_TYPES,
    }

    for column, valid_values in (
        category_rules.items()
    ):

        if column not in df.columns:
            continue

        actual_values = set(
            df[column]
            .dropna()
            .unique()
        )

        invalid_values = (
            actual_values - valid_values
        )

        if invalid_values:

            errors.append(
                f"{column}: invalid values "
                f"{sorted(invalid_values)}"
            )

    return errors


# =============================================================================
# DATA TYPE VALIDATION
# =============================================================================

def validate_data_types(
    df: pd.DataFrame,
) -> list[str]:
    """
    Validate expected numeric columns.
    """

    errors = []

    for column in EXPECTED_NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            errors.append(
                f"{column}: expected numeric type, "
                f"got {df[column].dtype}"
            )

    return errors


# =============================================================================
# SYNTHETIC GROUND-TRUTH VALIDATION
# =============================================================================

def validate_anomaly_labels(
    df: pd.DataFrame,
) -> list[str]:
    """
    Validate synthetic anomaly labels.

    This validation runs only when the complete
    ground-truth schema is present.
    """

    errors = []

    if not has_ground_truth(df):
        return errors

    anomaly_flag = (
        df["is_injected_anomaly"]
    )

    invalid_flags = ~anomaly_flag.isin(
        [True, False]
    )

    if invalid_flags.any():

        errors.append(
            "is_injected_anomaly contains "
            "invalid values."
        )

    inconsistent_normal = (
        (~anomaly_flag)
        & (
            df["anomaly_type"]
            != "none"
        )
    )

    if inconsistent_normal.any():

        errors.append(
            "Normal records contain "
            "an anomaly type."
        )

    inconsistent_anomaly = (
        anomaly_flag
        & (
            df["anomaly_type"]
            == "none"
        )
    )

    if inconsistent_anomaly.any():

        errors.append(
            "Anomalous records have "
            "anomaly_type='none'."
        )

    return errors


# =============================================================================
# COMPLETE DATASET VALIDATION
# =============================================================================

def validate_dataset(
    df: pd.DataFrame,
) -> dict:
    """
    Run the complete dataset validation pipeline.

    Ground-truth anomaly columns are optional.
    """

    errors = []

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    errors.extend(
        validate_schema(df)
    )

    # Stop early if required business columns
    # are missing.

    if errors:

        return {
            "valid": False,
            "errors": errors,
            "has_ground_truth": has_ground_truth(df),
        }

    # -------------------------------------------------------------------------
    # Standard validation
    # -------------------------------------------------------------------------

    errors.extend(
        validate_missing_values(df)
    )

    errors.extend(
        validate_duplicates(df)
    )

    errors.extend(
        validate_numeric_ranges(df)
    )

    errors.extend(
        validate_categories(df)
    )

    errors.extend(
        validate_data_types(df)
    )

    # -------------------------------------------------------------------------
    # Synthetic-only validation
    # -------------------------------------------------------------------------

    errors.extend(
        validate_anomaly_labels(df)
    )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "has_ground_truth": has_ground_truth(df),
    }


# =============================================================================
# DATASET PROFILE
# =============================================================================

def generate_profile(
    df: pd.DataFrame,
) -> dict:
    """
    Generate a lightweight dataset profile.

    Ground-truth anomaly statistics are included
    only when the dataset contains the synthetic
    labels.
    """

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isna()
            .sum()
            .sum()
        ),
        "duplicate_order_ids": int(
            df["order_id"]
            .duplicated()
            .sum()
        )
        if "order_id" in df.columns
        else 0,
        "has_ground_truth": has_ground_truth(df),
        "numeric_columns": [
            column
            for column in EXPECTED_NUMERIC_COLUMNS
            if column in df.columns
        ],
        "categorical_columns": [
            column
            for column in EXPECTED_CATEGORICAL_COLUMNS
            if column in df.columns
        ],
    }

    # -------------------------------------------------------------------------
    # Synthetic-only anomaly statistics
    # -------------------------------------------------------------------------

    if has_ground_truth(df):

        anomaly_count = int(
            df["is_injected_anomaly"]
            .astype(bool)
            .sum()
        )

        profile["anomaly_count"] = (
            anomaly_count
        )

        profile["anomaly_rate"] = (
            anomaly_count / len(df)
            if len(df) > 0
            else 0
        )

    else:

        profile["anomaly_count"] = None
        profile["anomaly_rate"] = None

    return profile


# =============================================================================
# HUMAN-READABLE VALIDATION REPORT
# =============================================================================

def print_validation_report(
    df: pd.DataFrame,
) -> None:
    """
    Print a human-readable validation report.
    """

    result = validate_dataset(df)
    profile = generate_profile(df)

    print()
    print("=" * 60)
    print(
        "INSIGHT — DATASET VALIDATION REPORT"
    )
    print("=" * 60)

    print(
        f"Rows:                "
        f"{profile['rows']:,}"
    )

    print(
        f"Columns:             "
        f"{profile['columns']}"
    )

    print(
        f"Missing values:      "
        f"{profile['missing_values']}"
    )

    print(
        f"Duplicate order IDs: "
        f"{profile['duplicate_order_ids']}"
    )

    print(
        f"Ground truth:        "
        f"{'Available' if profile['has_ground_truth'] else 'Not available'}"
    )

    if profile["has_ground_truth"]:

        print(
            f"Injected anomalies:  "
            f"{profile['anomaly_count']:,}"
        )

        print(
            f"Anomaly rate:        "
            f"{profile['anomaly_rate']:.2%}"
        )

    print()

    if result["valid"]:

        print("STATUS: PASS")
        print(
            "Dataset passed all validation checks."
        )

    else:

        print("STATUS: FAIL")
        print("Validation errors:")

        for error in result["errors"]:

            print(
                f"  - {error}"
            )

    print("=" * 60)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    dataset_path = (
        project_root
        / "data"
        / "synthetic_retail_data.csv"
    )

    df = load_dataset(
        dataset_path
    )

    print_validation_report(
        df
    )


if __name__ == "__main__":
    main()