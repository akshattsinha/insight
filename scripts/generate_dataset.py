from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
NUM_RECORDS = 10_000

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_retail_data.csv"


# ============================================================
# REFERENCE DATA
# ============================================================

PRODUCTS = {
    "Electronics": [
        ("Wireless Headphones", 3500),
        ("Smart Watch", 5000),
        ("Bluetooth Speaker", 2800),
        ("Power Bank", 1800),
        ("USB-C Hub", 2200),
    ],
    "Home": [
        ("Air Fryer", 6500),
        ("Coffee Maker", 4200),
        ("Mixer Grinder", 3800),
        ("Desk Lamp", 1600),
        ("Storage Organizer", 1200),
    ],
    "Fashion": [
        ("Running Shoes", 3200),
        ("Casual Shirt", 1800),
        ("Jeans", 2400),
        ("Backpack", 2100),
        ("Sports Jacket", 3500),
    ],
    "Beauty": [
        ("Face Serum", 1200),
        ("Moisturizer", 900),
        ("Shampoo", 700),
        ("Perfume", 2500),
        ("Hair Dryer", 2800),
    ],
    "Sports": [
        ("Yoga Mat", 1400),
        ("Dumbbell Set", 3500),
        ("Resistance Bands", 1000),
        ("Cricket Bat", 4200),
        ("Football", 1800),
    ],
}

REGIONS = [
    "North",
    "South",
    "East",
    "West",
    "Central",
]

PAYMENT_METHODS = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Cash on Delivery",
]

CUSTOMER_SEGMENTS = [
    "New",
    "Regular",
    "Premium",
]

CHANNELS = [
    "Website",
    "Mobile App",
    "Marketplace",
]

RETURN_REASONS = [
    "Damaged",
    "Wrong Item",
    "Quality Issue",
    "Changed Mind",
    "Late Delivery",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def choose_product(rng):
    category = rng.choice(list(PRODUCTS.keys()))

    products = PRODUCTS[category]
    product_index = rng.integers(0, len(products))

    product, base_price = products[product_index]

    return category, product, base_price


def generate_customer_segment(rng):
    return rng.choice(
        CUSTOMER_SEGMENTS,
        p=[0.35, 0.50, 0.15],
    )


def generate_discount(rng):
    return round(
        rng.choice(
            [0, 5, 10, 15, 20, 25, 30],
            p=[0.25, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05],
        ),
        2,
    )


# ============================================================
# DATA GENERATION
# ============================================================

def generate_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(
        start="2025-01-01",
        end="2025-12-31",
        freq="D",
    )

    records = []

    for index in range(NUM_RECORDS):

        order_date = rng.choice(dates)

        category, product, base_price = choose_product(rng)

        customer_segment = generate_customer_segment(rng)

        region = rng.choice(REGIONS)

        payment_method = rng.choice(
            PAYMENT_METHODS,
            p=[0.35, 0.20, 0.15, 0.10, 0.20],
        )

        channel = rng.choice(
            CHANNELS,
            p=[0.45, 0.35, 0.20],
        )

        quantity = int(
            rng.choice(
                [1, 2, 3, 4, 5],
                p=[0.45, 0.30, 0.15, 0.07, 0.03],
            )
        )

        discount_pct = generate_discount(rng)

        unit_price = round(
            base_price * rng.uniform(0.90, 1.10),
            2,
        )

        gross_revenue = unit_price * quantity

        discount_amount = (
            gross_revenue * discount_pct / 100
        )

        revenue = round(
            gross_revenue - discount_amount,
            2,
        )

        # Customer behavior influences returns.
        return_probability = 0.06

        if category == "Fashion":
            return_probability += 0.05

        if category == "Electronics":
            return_probability += 0.02

        if customer_segment == "New":
            return_probability += 0.02

        is_returned = (
            rng.random() < return_probability
        )

        return_amount = (
            revenue if is_returned else 0
        )

        # Delivery time depends on region/channel.
        delivery_days = rng.normal(
            loc=4.5,
            scale=1.4,
        )

        if region == "Central":
            delivery_days += 0.5

        if channel == "Marketplace":
            delivery_days += 0.7

        delivery_days = max(
            1,
            round(delivery_days),
        )

        # Customer satisfaction is influenced by
        # delivery performance and returns.
        satisfaction = (
            4.5
            - (delivery_days - 3) * 0.18
            - (1.0 if is_returned else 0)
            + rng.normal(0, 0.35)
        )

        satisfaction = float(
            np.clip(satisfaction, 1, 5)
        )

        # Operational cost has relationships with
        # quantity and delivery time.
        shipping_cost = (
            70
            + delivery_days * 18
            + quantity * 12
            + rng.normal(0, 15)
        )

        shipping_cost = max(
            30,
            round(shipping_cost, 2),
        )

        product_cost = (
            revenue * rng.uniform(0.48, 0.68)
        )

        operational_cost = round(
            product_cost + shipping_cost,
            2,
        )

        profit = round(
            revenue - operational_cost,
            2,
        )

        records.append(
            {
                "order_id": f"ORD-{index + 1:06d}",
                "order_date": order_date,
                "customer_id": f"CUST-{rng.integers(1, 2501):05d}",
                "region": region,
                "channel": channel,
                "customer_segment": customer_segment,
                "category": category,
                "product": product,
                "quantity": quantity,
                "unit_price": round(unit_price, 2),
                "discount_pct": discount_pct,
                "revenue": revenue,
                "return_status": (
                    "Returned"
                    if is_returned
                    else "Not Returned"
                ),
                "return_amount": round(return_amount, 2),
                "delivery_days": delivery_days,
                "satisfaction_score": round(
                    satisfaction,
                    2,
                ),
                "shipping_cost": shipping_cost,
                "operational_cost": operational_cost,
                "profit": profit,
                "payment_method": payment_method,

                # Ground truth fields.
                # These are ONLY for evaluating anomaly detection.
                "is_injected_anomaly": False,
                "anomaly_type": "none",
            }
        )

    df = pd.DataFrame(records)

    return df


# ============================================================
# CONTROLLED ANOMALIES
# ============================================================

def inject_anomalies(df, rng):
    anomaly_count = int(len(df) * 0.03)

    anomaly_indices = rng.choice(
        df.index,
        size=anomaly_count,
        replace=False,
    )

    anomaly_types = [
        "revenue_spike",
        "unusual_discount",
        "delivery_delay",
        "high_return",
        "profit_loss",
    ]

    for index in anomaly_indices:

        anomaly_type = rng.choice(
            anomaly_types
        )

        df.loc[
            index,
            "is_injected_anomaly",
        ] = True

        df.loc[
            index,
            "anomaly_type",
        ] = anomaly_type

        if anomaly_type == "revenue_spike":

            df.loc[
                index,
                "quantity",
            ] *= rng.integers(5, 10)

            df.loc[
                index,
                "revenue",
            ] = round(
                df.loc[index, "unit_price"]
                * df.loc[index, "quantity"]
                * (
                    1
                    - df.loc[index, "discount_pct"]
                    / 100
                ),
                2,
            )

            df.loc[
                index,
                "profit",
            ] = round(
                df.loc[index, "revenue"]
                - df.loc[index, "operational_cost"],
                2,
            )

        elif anomaly_type == "unusual_discount":

            df.loc[
                index,
                "discount_pct",
            ] = rng.choice(
                [60, 70, 80, 90]
            )

            df.loc[
                index,
                "discount_pct",
            ] = float(
                df.loc[index, "discount_pct"]
            )

            df.loc[
                index,
                "revenue",
            ] = round(
                df.loc[index, "unit_price"]
                * df.loc[index, "quantity"]
                * (
                    1
                    - df.loc[index, "discount_pct"]
                    / 100
                ),
                2,
            )

            df.loc[
                index,
                "profit",
            ] = round(
                df.loc[index, "revenue"]
                - df.loc[index, "operational_cost"],
                2,
            )

        elif anomaly_type == "delivery_delay":

            df.loc[
                index,
                "delivery_days",
            ] = int(
                rng.integers(15, 30)
            )

            df.loc[
                index,
                "satisfaction_score",
            ] = round(
                rng.uniform(1, 2.5),
                2,
            )

        elif anomaly_type == "high_return":

            df.loc[
                index,
                "return_status",
            ] = "Returned"

            df.loc[
                index,
                "return_amount",
            ] = df.loc[
                index,
                "revenue",
            ]

            df.loc[
                index,
                "satisfaction_score",
            ] = round(
                rng.uniform(1, 2.5),
                2,
            )

        elif anomaly_type == "profit_loss":

            df.loc[
                index,
                "operational_cost",
            ] = round(
                df.loc[index, "revenue"]
                * rng.uniform(1.5, 2.5),
                2,
            )

            df.loc[
                index,
                "profit",
            ] = round(
                df.loc[index, "revenue"]
                - df.loc[index, "operational_cost"],
                2,
            )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_generated_dataset(df):
    expected_columns = {
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
        "is_injected_anomaly",
        "anomaly_type",
    }

    missing_columns = (
        expected_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if len(df) != NUM_RECORDS:
        raise ValueError(
            f"Expected {NUM_RECORDS} rows, "
            f"got {len(df)}"
        )

    if df["order_id"].duplicated().any():
        raise ValueError(
            "Duplicate order IDs detected."
        )

    if df["revenue"].isna().any():
        raise ValueError(
            "Revenue contains missing values."
        )

    if df["quantity"].le(0).any():
        raise ValueError(
            "Quantity contains invalid values."
        )

    anomaly_rate = (
        df["is_injected_anomaly"].mean()
    )

    if not 0.02 <= anomaly_rate <= 0.04:
        raise ValueError(
            f"Unexpected anomaly rate: "
            f"{anomaly_rate:.2%}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("INSIGHT — Synthetic Dataset Generator")
    print("=" * 60)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print(
        f"Generating {NUM_RECORDS:,} records..."
    )

    df = generate_dataset()

    print("Injecting controlled anomalies...")

    df = inject_anomalies(
        df,
        rng,
    )

    print("Validating generated dataset...")

    validate_generated_dataset(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    anomaly_count = int(
        df["is_injected_anomaly"].sum()
    )

    print()
    print("Dataset generated successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Injected anomalies: {anomaly_count:,}"
    )
    print(
        f"Anomaly rate: "
        f"{anomaly_count / len(df):.2%}"
    )
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()