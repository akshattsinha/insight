from pathlib import Path

import pandas as pd

from analysis.validator import load_dataset


def calculate_total_revenue(
    df: pd.DataFrame,
) -> float:
    """
    Calculate total revenue.
    """

    return float(df["revenue"].sum())


def calculate_total_orders(
    df: pd.DataFrame,
) -> int:
    """
    Calculate the number of unique orders.
    """

    return int(df["order_id"].nunique())


def calculate_total_customers(
    df: pd.DataFrame,
) -> int:
    """
    Calculate the number of unique customers.
    """

    return int(df["customer_id"].nunique())


def calculate_total_profit(
    df: pd.DataFrame,
) -> float:
    """
    Calculate total profit.
    """

    return float(df["profit"].sum())


def calculate_profit_margin(
    df: pd.DataFrame,
) -> float:
    """
    Calculate profit margin as a percentage.
    """

    revenue = calculate_total_revenue(df)
    profit = calculate_total_profit(df)

    if revenue == 0:
        return 0.0

    return float(
        (profit / revenue) * 100
    )


def calculate_average_order_value(
    df: pd.DataFrame,
) -> float:
    """
    Calculate average revenue per order.
    """

    revenue = calculate_total_revenue(df)
    orders = calculate_total_orders(df)

    if orders == 0:
        return 0.0

    return float(
        revenue / orders
    )


def calculate_return_rate(
    df: pd.DataFrame,
) -> float:
    """
    Calculate the percentage of orders that
    were returned.
    """

    orders = calculate_total_orders(df)

    if orders == 0:
        return 0.0

    returned_orders = int(
        (df["return_status"] == "Returned")
        .sum()
    )

    return float(
        (returned_orders / orders) * 100
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
    Calculate all primary INSIGHT KPIs.
    """

    return {
        "total_revenue": calculate_total_revenue(df),
        "total_orders": calculate_total_orders(df),
        "total_customers": calculate_total_customers(df),
        "total_profit": calculate_total_profit(df),
        "profit_margin": calculate_profit_margin(df),
        "average_order_value": (
            calculate_average_order_value(df)
        ),
        "return_rate": calculate_return_rate(df),
        "average_delivery_time": (
            calculate_average_delivery_time(df)
        ),
        "average_satisfaction": (
            calculate_average_satisfaction(df)
        ),
    }


def print_kpis(
    kpis: dict,
) -> None:
    """
    Print KPIs in a human-readable format.
    """

    print()
    print("=" * 60)
    print("INSIGHT — KPI REPORT")
    print("=" * 60)

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

    kpis = calculate_kpis(df)

    print_kpis(kpis)


if __name__ == "__main__":
    main()