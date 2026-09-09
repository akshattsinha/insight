from pathlib import Path

import pandas as pd

from analysis.validator import load_dataset


def prepare_date_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert order_date into a datetime column
    and create a monthly period column.
    """

    result = df.copy()

    result["order_date"] = pd.to_datetime(
        result["order_date"],
        errors="coerce",
    )

    if result["order_date"].isna().any():
        raise ValueError(
            "Invalid dates found in order_date."
        )

    result["month"] = (
        result["order_date"]
        .dt.to_period("M")
    )

    return result


def calculate_monthly_trends(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate monthly business performance metrics.
    """

    data = prepare_date_column(df)

    monthly = (
        data
        .groupby("month")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_id", "nunique"),
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

    monthly["return_rate"] = (
        monthly["returns"]
        / monthly["orders"]
        * 100
    )

    monthly["profit_margin"] = (
        monthly["profit"]
        / monthly["revenue"]
        * 100
    )

    monthly["average_order_value"] = (
        monthly["revenue"]
        / monthly["orders"]
    )

    monthly["month"] = (
        monthly["month"]
        .astype(str)
    )

    return monthly


def calculate_month_over_month_changes(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate month-over-month percentage changes
    for key business metrics.
    """

    result = monthly.copy()

    metrics = [
        "revenue",
        "profit",
        "orders",
        "customers",
        "return_rate",
        "average_delivery_days",
        "average_satisfaction",
        "profit_margin",
        "average_order_value",
    ]

    for metric in metrics:

        result[
            f"{metric}_mom_change"
        ] = (
            result[metric]
            .pct_change()
            * 100
        )

    return result


def calculate_trends(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the complete monthly trend analysis.
    """

    monthly = calculate_monthly_trends(df)

    return calculate_month_over_month_changes(
        monthly
    )


def find_extreme_months(
    trends: pd.DataFrame,
) -> dict:
    """
    Identify the strongest and weakest months
    for major business metrics.
    """

    result = {}

    metrics = [
        "revenue",
        "profit",
        "orders",
        "return_rate",
        "average_delivery_days",
        "average_satisfaction",
    ]

    for metric in metrics:

        highest_index = trends[
            metric
        ].idxmax()

        lowest_index = trends[
            metric
        ].idxmin()

        result[metric] = {
            "highest": {
                "month": trends.loc[
                    highest_index,
                    "month",
                ],
                "value": float(
                    trends.loc[
                        highest_index,
                        metric,
                    ]
                ),
            },
            "lowest": {
                "month": trends.loc[
                    lowest_index,
                    "month",
                ],
                "value": float(
                    trends.loc[
                        lowest_index,
                        metric,
                    ]
                ),
            },
        }

    return result


def print_trend_report(
    trends: pd.DataFrame,
) -> None:
    """
    Print a human-readable monthly trend report.
    """

    print()
    print("=" * 100)
    print("INSIGHT — MONTHLY TREND ANALYSIS")
    print("=" * 100)

    print()

    print(
        f"{'Month':<10}"
        f"{'Revenue':>15}"
        f"{'Profit':>15}"
        f"{'Orders':>10}"
        f"{'Returns':>10}"
        f"{'Return %':>10}"
        f"{'Delivery':>10}"
        f"{'Satisfaction':>13}"
    )

    print("-" * 100)

    for _, row in trends.iterrows():

        print(
            f"{row['month']:<10}"
            f"₹{row['revenue']:>13,.0f}"
            f"₹{row['profit']:>13,.0f}"
            f"{row['orders']:>10,.0f}"
            f"{row['returns']:>10,.0f}"
            f"{row['return_rate']:>9.2f}%"
            f"{row['average_delivery_days']:>9.2f}"
            f"{row['average_satisfaction']:>12.2f}"
        )

    print()

    print("MONTH-OVER-MONTH REVENUE CHANGES")

    print("-" * 50)

    for _, row in trends.iterrows():

        change = row[
            "revenue_mom_change"
        ]

        if pd.isna(change):
            print(
                f"{row['month']}: N/A"
            )
        else:
            print(
                f"{row['month']}: "
                f"{change:+.2f}%"
            )

    extremes = find_extreme_months(
        trends
    )

    print()

    print("EXTREME MONTHS")

    print("-" * 50)

    for metric, values in extremes.items():

        print(
            f"{metric}:"
        )

        print(
            f"  Highest → "
            f"{values['highest']['month']} "
            f"({values['highest']['value']:,.2f})"
        )

        print(
            f"  Lowest  → "
            f"{values['lowest']['month']} "
            f"({values['lowest']['value']:,.2f})"
        )

    print("=" * 100)


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

    trends = calculate_trends(df)

    print_trend_report(trends)


if __name__ == "__main__":
    main()