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


SEGMENT_COLUMNS = [
    "category",
    "region",
    "channel",
    "customer_segment",
    "payment_method",
]


def calculate_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Pearson correlations between
    numerical business variables.
    """

    available_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    return df[
        available_columns
    ].corr()


def extract_strong_correlations(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.40,
) -> list[dict]:
    """
    Extract unique variable pairs whose absolute
    correlation is at or above the threshold.

    Correlation does not imply causation.
    """

    results = []

    columns = correlation_matrix.columns

    for i, column_a in enumerate(columns):

        for j in range(i + 1, len(columns)):

            column_b = columns[j]

            correlation = correlation_matrix.loc[
                column_a,
                column_b,
            ]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= threshold:

                direction = (
                    "positive"
                    if correlation > 0
                    else "negative"
                )

                results.append(
                    {
                        "variable_a": column_a,
                        "variable_b": column_b,
                        "correlation": round(
                            float(correlation),
                            3,
                        ),
                        "direction": direction,
                    }
                )

    results.sort(
        key=lambda item: abs(
            item["correlation"]
        ),
        reverse=True,
    )

    return results


def calculate_segment_performance(
    df: pd.DataFrame,
    segment_column: str,
) -> pd.DataFrame:
    """
    Calculate business performance by a
    categorical segment.
    """

    grouped = (
        df.groupby(segment_column)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique"),
            returns=(
                "return_status",
                lambda values: (
                    values == "Returned"
                ).sum(),
            ),
            average_delivery_days=(
                "delivery_days",
                "mean",
            ),
            average_satisfaction=(
                "satisfaction_score",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["return_rate"] = (
        grouped["returns"]
        / grouped["orders"]
        * 100
    )

    grouped["profit_margin"] = (
        grouped["profit"]
        / grouped["revenue"]
        * 100
    )

    grouped["average_order_value"] = (
        grouped["revenue"]
        / grouped["orders"]
    )

    return grouped


def calculate_all_segment_performance(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Calculate segment-level performance for
    all configured categorical dimensions.
    """

    results = {}

    for column in SEGMENT_COLUMNS:

        if column not in df.columns:
            continue

        results[column] = (
            calculate_segment_performance(
                df,
                column,
            )
        )

    return results


def find_segment_extremes(
    segment_results: dict[str, pd.DataFrame],
) -> dict:
    """
    Identify strongest and weakest segments
    across important business metrics.
    """

    result = {}

    metrics = [
        "revenue",
        "profit",
        "return_rate",
        "average_delivery_days",
        "average_satisfaction",
        "profit_margin",
    ]

    for segment, dataframe in (
        segment_results.items()
    ):

        segment_extremes = {}

        for metric in metrics:

            highest_index = dataframe[
                metric
            ].idxmax()

            lowest_index = dataframe[
                metric
            ].idxmin()

            segment_extremes[metric] = {
                "highest": {
                    "segment": str(
                        dataframe.loc[
                            highest_index,
                            segment,
                        ]
                    ),
                    "value": float(
                        dataframe.loc[
                            highest_index,
                            metric,
                        ]
                    ),
                },
                "lowest": {
                    "segment": str(
                        dataframe.loc[
                            lowest_index,
                            segment,
                        ]
                    ),
                    "value": float(
                        dataframe.loc[
                            lowest_index,
                            metric,
                        ]
                    ),
                },
            }

        result[segment] = segment_extremes

    return result


def calculate_patterns(
    df: pd.DataFrame,
) -> dict:
    """
    Generate the complete correlation and
    segment-level pattern analysis.
    """

    correlation_matrix = (
        calculate_correlations(df)
    )

    strong_correlations = (
        extract_strong_correlations(
            correlation_matrix
        )
    )

    segment_results = (
        calculate_all_segment_performance(df)
    )

    segment_extremes = find_segment_extremes(
        segment_results
    )

    return {
        "correlation_matrix": correlation_matrix,
        "strong_correlations": strong_correlations,
        "segment_performance": segment_results,
        "segment_extremes": segment_extremes,
    }


def print_correlation_report(
    patterns: dict,
) -> None:
    """
    Print strong correlations.
    """

    print()
    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    correlations = patterns[
        "strong_correlations"
    ]

    if not correlations:

        print(
            "No correlations exceeded "
            "the configured threshold."
        )

    else:

        print(
            f"{'Variable A':<25}"
            f"{'Variable B':<25}"
            f"{'Correlation':>12}"
            f"{'Direction':>12}"
        )

        print("-" * 80)

        for item in correlations:

            print(
                f"{item['variable_a']:<25}"
                f"{item['variable_b']:<25}"
                f"{item['correlation']:>12.3f}"
                f"{item['direction']:>12}"
            )

    print()


def print_segment_reports(
    patterns: dict,
) -> None:
    """
    Print segment-level business performance.
    """

    segment_results = patterns[
        "segment_performance"
    ]

    for segment, dataframe in (
        segment_results.items()
    ):

        print("=" * 80)
        print(
            f"PERFORMANCE BY {segment.upper()}"
        )
        print("=" * 80)

        print(
            f"{segment:<20}"
            f"{'Revenue':>15}"
            f"{'Profit':>15}"
            f"{'Orders':>10}"
            f"{'Return %':>10}"
            f"{'Delivery':>10}"
            f"{'Satisfaction':>13}"
        )

        print("-" * 80)

        for _, row in dataframe.iterrows():

            print(
                f"{str(row[segment]):<20}"
                f"₹{row['revenue']:>13,.0f}"
                f"₹{row['profit']:>13,.0f}"
                f"{row['orders']:>10,.0f}"
                f"{row['return_rate']:>9.2f}%"
                f"{row['average_delivery_days']:>9.2f}"
                f"{row['average_satisfaction']:>12.2f}"
            )

        print()


def print_extreme_reports(
    patterns: dict,
) -> None:
    """
    Print strongest and weakest segments.
    """

    extremes = patterns[
        "segment_extremes"
    ]

    print("=" * 80)
    print("SEGMENT EXTREMES")
    print("=" * 80)

    for segment, metrics in extremes.items():

        print()
        print(
            f"{segment.upper()}"
        )

        for metric, values in metrics.items():

            highest = values["highest"]
            lowest = values["lowest"]

            print(
                f"  {metric}:"
            )

            print(
                f"    Highest → "
                f"{highest['segment']} "
                f"({highest['value']:,.2f})"
            )

            print(
                f"    Lowest  → "
                f"{lowest['segment']} "
                f"({lowest['value']:,.2f})"
            )


def print_pattern_report(
    patterns: dict,
) -> None:
    """
    Print the complete pattern analysis report.
    """

    print_correlation_report(
        patterns
    )

    print_segment_reports(
        patterns
    )

    print_extreme_reports(
        patterns
    )

    print()
    print("=" * 80)
    print(
        "NOTE: Correlations indicate association, "
        "not causation."
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

    df = load_dataset(dataset_path)

    patterns = calculate_patterns(df)

    print_pattern_report(patterns)


if __name__ == "__main__":
    main()