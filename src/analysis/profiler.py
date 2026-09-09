from pathlib import Path

import pandas as pd

from analysis.validator import load_dataset


NUMERIC_COLUMNS = [
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


CATEGORICAL_COLUMNS = [
    "region",
    "channel",
    "customer_segment",
    "category",
    "product",
    "return_status",
    "payment_method",
]


def profile_shape(df: pd.DataFrame) -> dict:
    """
    Return basic dataset dimensions.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
    }


def profile_data_types(
    df: pd.DataFrame,
) -> dict:
    """
    Return the data type of every column.
    """

    return {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }


def profile_missing_values(
    df: pd.DataFrame,
) -> dict:
    """
    Return missing-value counts and percentages.
    """

    counts = df.isna().sum()

    percentages = (
        counts / len(df) * 100
        if len(df) > 0
        else counts
    )

    return {
        column: {
            "count": int(counts[column]),
            "percentage": round(
                float(percentages[column]),
                2,
            ),
        }
        for column in df.columns
    }


def profile_numeric_columns(
    df: pd.DataFrame,
) -> dict:
    """
    Generate descriptive statistics for
    numerical business variables.
    """

    available_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    if not available_columns:
        return {}

    statistics = df[
        available_columns
    ].describe().T

    result = {}

    for column in statistics.index:

        result[column] = {
            "count": int(
                statistics.loc[column, "count"]
            ),
            "mean": round(
                float(statistics.loc[column, "mean"]),
                2,
            ),
            "std": round(
                float(statistics.loc[column, "std"]),
                2,
            ),
            "min": round(
                float(statistics.loc[column, "min"]),
                2,
            ),
            "25%": round(
                float(statistics.loc[column, "25%"]),
                2,
            ),
            "median": round(
                float(statistics.loc[column, "50%"]),
                2,
            ),
            "75%": round(
                float(statistics.loc[column, "75%"]),
                2,
            ),
            "max": round(
                float(statistics.loc[column, "max"]),
                2,
            ),
        }

    return result


def profile_categorical_columns(
    df: pd.DataFrame,
) -> dict:
    """
    Generate frequency distributions for
    categorical business variables.
    """

    result = {}

    for column in CATEGORICAL_COLUMNS:

        if column not in df.columns:
            continue

        value_counts = (
            df[column]
            .value_counts(dropna=False)
        )

        result[column] = {
            str(value): int(count)
            for value, count
            in value_counts.items()
        }

    return result


def profile_date_column(
    df: pd.DataFrame,
) -> dict:
    """
    Profile the order date range.
    """

    dates = pd.to_datetime(
        df["order_date"],
        errors="coerce",
    )

    return {
        "minimum": str(dates.min().date()),
        "maximum": str(dates.max().date()),
        "unique_days": int(dates.dt.date.nunique()),
    }


def profile_correlations(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate correlations between numerical
    business variables.

    Correlation indicates association, not causation.
    """

    available_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    correlation_matrix = df[
        available_columns
    ].corr()

    result = {}

    for column in correlation_matrix.columns:

        result[column] = {
            other_column: round(
                float(
                    correlation_matrix.loc[
                        column,
                        other_column,
                    ]
                ),
                3,
            )
            for other_column
            in correlation_matrix.columns
        }

    return result


def profile_anomalies(
    df: pd.DataFrame,
) -> dict:
    """
    Profile the ground-truth anomaly labels.

    These labels are for evaluation only.
    """

    anomaly_count = int(
        df["is_injected_anomaly"].sum()
    )

    normal_count = int(
        (~df["is_injected_anomaly"]).sum()
    )

    anomaly_types = (
        df.loc[
            df["is_injected_anomaly"],
            "anomaly_type",
        ]
        .value_counts()
        .to_dict()
    )

    return {
        "normal_records": normal_count,
        "anomalous_records": anomaly_count,
        "anomaly_rate": round(
            anomaly_count / len(df),
            4,
        ),
        "anomaly_types": {
            str(key): int(value)
            for key, value
            in anomaly_types.items()
        },
    }


def generate_profile(
    df: pd.DataFrame,
) -> dict:
    """
    Generate the complete dataset profile.
    """

    return {
        "shape": profile_shape(df),
        "data_types": profile_data_types(df),
        "missing_values": profile_missing_values(df),
        "numeric_statistics": profile_numeric_columns(df),
        "categorical_distributions": (
            profile_categorical_columns(df)
        ),
        "date_profile": profile_date_column(df),
        "correlations": profile_correlations(df),
        "anomalies": profile_anomalies(df),
    }


def print_profile(
    profile: dict,
) -> None:
    """
    Print a concise human-readable profile.
    """

    shape = profile["shape"]
    date_profile = profile["date_profile"]
    anomalies = profile["anomalies"]

    print()
    print("=" * 60)
    print("INSIGHT — DATASET PROFILE")
    print("=" * 60)

    print(
        f"Rows:              {shape['rows']:,}"
    )

    print(
        f"Columns:           {shape['columns']}"
    )

    print(
        f"Date range:        "
        f"{date_profile['minimum']} → "
        f"{date_profile['maximum']}"
    )

    print(
        f"Unique days:       "
        f"{date_profile['unique_days']:,}"
    )

    print()

    print("NUMERICAL VARIABLES")

    for column, statistics in (
        profile["numeric_statistics"].items()
    ):
        print(
            f"  {column:22} "
            f"mean={statistics['mean']:>10,.2f} "
            f"median={statistics['median']:>10,.2f} "
            f"min={statistics['min']:>10,.2f} "
            f"max={statistics['max']:>10,.2f}"
        )

    print()

    print("CATEGORICAL VARIABLES")

    for column, distribution in (
        profile[
            "categorical_distributions"
        ].items()
    ):

        top_values = list(
            distribution.items()
        )[:5]

        formatted = ", ".join(
            f"{value}: {count:,}"
            for value, count
            in top_values
        )

        print(
            f"  {column:20} {formatted}"
        )

    print()

    print("ANOMALIES — GROUND TRUTH")

    print(
        f"  Normal records:   "
        f"{anomalies['normal_records']:,}"
    )

    print(
        f"  Anomalous records:"
        f" {anomalies['anomalous_records']:,}"
    )

    print(
        f"  Anomaly rate:     "
        f"{anomalies['anomaly_rate']:.2%}"
    )

    print("  Types:")

    for anomaly_type, count in (
        anomalies["anomaly_types"].items()
    ):
        print(
            f"    {anomaly_type:20} "
            f"{count:,}"
        )

    print("=" * 60)


def main():
    project_root = (
        Path(__file__).resolve().parents[2]
    )

    dataset_path = (
        project_root
        / "data"
        / "synthetic_retail_data.csv"
    )

    df = load_dataset(dataset_path)

    profile = generate_profile(df)

    print_profile(profile)


if __name__ == "__main__":
    main()