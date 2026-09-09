import sys
from pathlib import Path

import pandas as pd


# Add src directory to Python's import path.
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from analysis.validator import load_dataset


def calculate_total_revenue(
    df: pd.DataFrame,
) -> float:
    """
    Calculate total revenue.
    """

    return float(
        df["revenue"].sum()
    )


def calculate_total_orders(
    df: pd.DataFrame,
) -> int:
    """
    Calculate total number of unique orders.
    """

    return int(
        df["order_id"].nunique()
    )


def calculate_total_customers(
    df: pd.DataFrame,
) -> int:
    """
    Calculate total number of unique customers.
    """

    return int(
        df["customer_id"].nunique()
    )


def calculate_total_profit(
    df: pd.DataFrame,
) -> float:
    """
    Calculate total profit.
    """

    return float(
        df["profit"].sum()
    )


def calculate_profit_margin(
    total_profit: float,
    total_revenue: float,
) -> float:
    """
    Calculate profit margin as a percentage.
    """

    if total_revenue == 0:
        return 0.0

    return (
        total_profit
        / total_revenue
        * 100
    )


def calculate_average_order_value(
    total_revenue: float,
    total_orders: int,
) -> float:
    """
    Calculate average order value.
    """

    if total_orders == 0:
        return 0.0

    return (
        total_revenue
        / total_orders
    )


def calculate_return_rate(
    df: pd.DataFrame,
    total_orders: int,
) -> float:
    """
    Calculate the percentage of returned orders.
    """

    if total_orders == 0:
        return 0.0

    returned_orders = int(
        (
            df["return_status"]
            == "Returned"
        ).sum()
    )

    return (
        returned_orders
        / total_orders
        * 100
    )


def calculate_average_delivery_time(
    df: pd.DataFrame,
) -> float:
    """
    Calculate average delivery time in days.
    """

    return float(
        df["delivery_days"].mean()
    )


def calculate_average_satisfaction(
    df: pd.DataFrame,
) -> float:
    """
    Calculate average customer satisfaction.
    """

    return float(
        df["satisfaction_score"].mean()
    )


def calculate_kpis(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate the complete set of core
    business KPIs for INSIGHT.
    """

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    required_columns = [
        "revenue",
        "profit",
        "order_id",
        "customer_id",
        "return_status",
        "delivery_days",
        "satisfaction_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    total_revenue = (
        calculate_total_revenue(df)
    )

    total_orders = (
        calculate_total_orders(df)
    )

    total_customers = (
        calculate_total_customers(df)
    )

    total_profit = (
        calculate_total_profit(df)
    )

    profit_margin = (
        calculate_profit_margin(
            total_profit,
            total_revenue,
        )
    )

    average_order_value = (
        calculate_average_order_value(
            total_revenue,
            total_orders,
        )
    )

    return_rate = (
        calculate_return_rate(
            df,
            total_orders,
        )
    )

    average_delivery_time = (
        calculate_average_delivery_time(df)
    )

    average_satisfaction = (
        calculate_average_satisfaction(df)
    )

    return {
        "total_revenue": round(
            total_revenue,
            2,
        ),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "total_profit": round(
            total_profit,
            2,
        ),
        "profit_margin": round(
            profit_margin,
            2,
        ),
        "average_order_value": round(
            average_order_value,
            2,
        ),
        "return_rate": round(
            return_rate,
            2,
        ),
        "average_delivery_time": round(
            average_delivery_time,
            2,
        ),
        "average_satisfaction": round(
            average_satisfaction,
            2,
        ),
    }


def print_kpi_report(
    kpis: dict,
) -> None:
    """
    Print a human-readable KPI report.
    """

    print()
    print("=" * 80)
    print("INSIGHT — BUSINESS KPI SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Total Revenue:          "
        f"₹{kpis['total_revenue']:,.2f}"
    )

    print(
        f"Total Orders:           "
        f"{kpis['total_orders']:,}"
    )

    print(
        f"Total Customers:        "
        f"{kpis['total_customers']:,}"
    )

    print(
        f"Total Profit:           "
        f"₹{kpis['total_profit']:,.2f}"
    )

    print(
        f"Profit Margin:          "
        f"{kpis['profit_margin']:.2f}%"
    )

    print(
        f"Average Order Value:    "
        f"₹{kpis['average_order_value']:,.2f}"
    )

    print(
        f"Return Rate:            "
        f"{kpis['return_rate']:.2f}%"
    )

    print(
        f"Average Delivery Time:  "
        f"{kpis['average_delivery_time']:.2f} days"
    )

    print(
        f"Average Satisfaction:   "
        f"{kpis['average_satisfaction']:.2f}/5"
    )

    print()


def main() -> None:
    """
    Load the synthetic dataset and
    print the KPI report.
    """

    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "synthetic_retail_data.csv"
    )

    df = load_dataset(
        dataset_path
    )

    kpis = calculate_kpis(df)

    print_kpi_report(kpis)


if __name__ == "__main__":
    main()