import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

# Add src directory to Python's import path.
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.validator import load_dataset
from analysis.kpis import calculate_kpis
from analysis.trends import calculate_monthly_trends
from analysis.patterns import calculate_patterns
from features.engineering import engineer_features
from models.anomaly_detector import AnomalyDetector


@dataclass
class Insight:
    """
    Structured business insight generated from
    verified analytical evidence.
    """

    title: str
    finding: str
    evidence: list[dict[str, Any]]
    severity: str
    impact: str
    possible_contributing_factors: list[str]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_insight(
    title: str,
    finding: str,
    evidence: list[dict[str, Any]],
    severity: str,
    impact: str,
    possible_contributing_factors: list[str],
    recommendation: str,
) -> Insight:
    """
    Create a validated structured insight.
    """

    valid_severities = {
        "low",
        "medium",
        "high",
        "critical",
    }

    if severity not in valid_severities:
        raise ValueError(
            f"Invalid severity: {severity}"
        )

    if not evidence:
        raise ValueError(
            "An insight must contain evidence."
        )

    return Insight(
        title=title,
        finding=finding,
        evidence=evidence,
        severity=severity,
        impact=impact,
        possible_contributing_factors=(
            possible_contributing_factors
        ),
        recommendation=recommendation,
    )


def generate_trend_insights(
    monthly_metrics: pd.DataFrame,
) -> list[Insight]:
    """
    Generate insights from monthly business trends.
    """

    insights = []

    if len(monthly_metrics) < 2:
        return insights

    latest = monthly_metrics.iloc[-1]
    previous = monthly_metrics.iloc[-2]

    latest_revenue = float(
        latest["revenue"]
    )

    previous_revenue = float(
        previous["revenue"]
    )

    if previous_revenue != 0:

        revenue_change = (
            (latest_revenue - previous_revenue)
            / previous_revenue
            * 100
        )

        if abs(revenue_change) >= 5:

            direction = (
                "increased"
                if revenue_change > 0
                else "decreased"
            )

            severity = (
                "high"
                if abs(revenue_change) >= 10
                else "medium"
            )

            insights.append(
                create_insight(
                    title="Monthly revenue change",
                    finding=(
                        f"Revenue {direction} by "
                        f"{abs(revenue_change):.2f}% "
                        f"from "
                        f"{previous['month']} to "
                        f"{latest['month']}."
                    ),
                    evidence=[
                        {
                            "metric": "revenue",
                            "previous_month": str(
                                previous["month"]
                            ),
                            "previous_value": round(
                                previous_revenue,
                                2,
                            ),
                            "current_month": str(
                                latest["month"]
                            ),
                            "current_value": round(
                                latest_revenue,
                                2,
                            ),
                            "change_pct": round(
                                revenue_change,
                                2,
                            ),
                        }
                    ],
                    severity=severity,
                    impact=(
                        "A sustained revenue change can "
                        "materially affect overall business "
                        "performance."
                    ),
                    possible_contributing_factors=[
                        "Changes in order volume",
                        "Changes in average order value",
                        "Changes in customer or channel mix",
                    ],
                    recommendation=(
                        "Investigate order volume, customer "
                        "segments, categories, and channels "
                        "for the period before taking action."
                    ),
                )
            )

    return insights


def generate_segment_insights(
    patterns: dict,
) -> list[Insight]:
    """
    Generate insights from segment-level analysis.
    """

    insights = []

    # ---------------------------------------------------------
    # Category return rate
    # ---------------------------------------------------------

    category_data = patterns[
        "segment_performance"
    ].get("category")

    if category_data is not None:

        highest_return_index = category_data[
            "return_rate"
        ].idxmax()

        lowest_return_index = category_data[
            "return_rate"
        ].idxmin()

        highest_return_category = str(
            category_data.loc[
                highest_return_index,
                "category",
            ]
        )

        highest_return_rate = float(
            category_data.loc[
                highest_return_index,
                "return_rate",
            ]
        )

        lowest_return_category = str(
            category_data.loc[
                lowest_return_index,
                "category",
            ]
        )

        lowest_return_rate = float(
            category_data.loc[
                lowest_return_index,
                "return_rate",
            ]
        )

        return_gap = (
            highest_return_rate
            - lowest_return_rate
        )

        if return_gap >= 3:

            insights.append(
                create_insight(
                    title=(
                        "High-return category identified"
                    ),
                    finding=(
                        f"{highest_return_category} has "
                        f"the highest return rate at "
                        f"{highest_return_rate:.2f}%, "
                        f"compared with "
                        f"{lowest_return_category} at "
                        f"{lowest_return_rate:.2f}%."
                    ),
                    evidence=[
                        {
                            "metric": "return_rate",
                            "highest_segment": (
                                highest_return_category
                            ),
                            "highest_value_pct": round(
                                highest_return_rate,
                                2,
                            ),
                            "lowest_segment": (
                                lowest_return_category
                            ),
                            "lowest_value_pct": round(
                                lowest_return_rate,
                                2,
                            ),
                            "gap_percentage_points": round(
                                return_gap,
                                2,
                            ),
                        }
                    ],
                    severity=(
                        "high"
                        if return_gap >= 5
                        else "medium"
                    ),
                    impact=(
                        "Higher returns can increase "
                        "reverse-logistics costs and "
                        "reduce realized profitability."
                    ),
                    possible_contributing_factors=[
                        (
                            "Product or category-level "
                            "customer expectations"
                        ),
                        (
                            "Delivery or fulfillment "
                            "experience"
                        ),
                        (
                            "Product-specific return "
                            "behavior"
                        ),
                    ],
                    recommendation=(
                        f"Investigate {highest_return_category} "
                        "at the product and operational level "
                        "to identify the drivers of its elevated "
                        "return rate."
                    ),
                )
            )

    # ---------------------------------------------------------
    # Region delivery performance
    # ---------------------------------------------------------

    region_data = patterns[
        "segment_performance"
    ].get("region")

    if region_data is not None:

        highest_delivery_index = region_data[
            "average_delivery_days"
        ].idxmax()

        lowest_delivery_index = region_data[
            "average_delivery_days"
        ].idxmin()

        slowest_region = str(
            region_data.loc[
                highest_delivery_index,
                "region",
            ]
        )

        slowest_delivery = float(
            region_data.loc[
                highest_delivery_index,
                "average_delivery_days",
            ]
        )

        fastest_region = str(
            region_data.loc[
                lowest_delivery_index,
                "region",
            ]
        )

        fastest_delivery = float(
            region_data.loc[
                lowest_delivery_index,
                "average_delivery_days",
            ]
        )

        delivery_gap = (
            slowest_delivery
            - fastest_delivery
        )

        if delivery_gap >= 0.25:

            insights.append(
                create_insight(
                    title=(
                        "Regional delivery disparity"
                    ),
                    finding=(
                        f"{slowest_region} has the highest "
                        f"average delivery time at "
                        f"{slowest_delivery:.2f} days, "
                        f"compared with "
                        f"{fastest_region} at "
                        f"{fastest_delivery:.2f} days."
                    ),
                    evidence=[
                        {
                            "metric": (
                                "average_delivery_days"
                            ),
                            "slowest_region": (
                                slowest_region
                            ),
                            "slowest_value_days": round(
                                slowest_delivery,
                                2,
                            ),
                            "fastest_region": (
                                fastest_region
                            ),
                            "fastest_value_days": round(
                                fastest_delivery,
                                2,
                            ),
                            "gap_days": round(
                                delivery_gap,
                                2,
                            ),
                        }
                    ],
                    severity=(
                        "high"
                        if delivery_gap >= 0.5
                        else "medium"
                    ),
                    impact=(
                        "Longer delivery times may increase "
                        "customer dissatisfaction and "
                        "operational friction."
                    ),
                    possible_contributing_factors=[
                        "Regional logistics constraints",
                        "Fulfillment differences",
                        "Shipping distance or capacity",
                    ],
                    recommendation=(
                        f"Investigate fulfillment and "
                        f"logistics performance in "
                        f"{slowest_region}."
                    ),
                )
            )

    # ---------------------------------------------------------
    # Channel satisfaction
    # ---------------------------------------------------------

    channel_data = patterns[
        "segment_performance"
    ].get("channel")

    if channel_data is not None:

        lowest_satisfaction_index = channel_data[
            "average_satisfaction"
        ].idxmin()

        highest_satisfaction_index = channel_data[
            "average_satisfaction"
        ].idxmax()

        lowest_channel = str(
            channel_data.loc[
                lowest_satisfaction_index,
                "channel",
            ]
        )

        lowest_satisfaction = float(
            channel_data.loc[
                lowest_satisfaction_index,
                "average_satisfaction",
            ]
        )

        highest_channel = str(
            channel_data.loc[
                highest_satisfaction_index,
                "channel",
            ]
        )

        highest_satisfaction = float(
            channel_data.loc[
                highest_satisfaction_index,
                "average_satisfaction",
            ]
        )

        satisfaction_gap = (
            highest_satisfaction
            - lowest_satisfaction
        )

        if satisfaction_gap >= 0.05:

            insights.append(
                create_insight(
                    title=(
                        "Channel satisfaction disparity"
                    ),
                    finding=(
                        f"{lowest_channel} has the lowest "
                        f"average satisfaction at "
                        f"{lowest_satisfaction:.2f}/5, "
                        f"while {highest_channel} has "
                        f"{highest_satisfaction:.2f}/5."
                    ),
                    evidence=[
                        {
                            "metric": (
                                "average_satisfaction"
                            ),
                            "lowest_channel": (
                                lowest_channel
                            ),
                            "lowest_value": round(
                                lowest_satisfaction,
                                2,
                            ),
                            "highest_channel": (
                                highest_channel
                            ),
                            "highest_value": round(
                                highest_satisfaction,
                                2,
                            ),
                            "gap": round(
                                satisfaction_gap,
                                2,
                            ),
                        }
                    ],
                    severity="medium",
                    impact=(
                        "Lower satisfaction in a channel "
                        "may indicate an opportunity to "
                        "improve the customer experience."
                    ),
                    possible_contributing_factors=[
                        "Delivery experience",
                        "Channel-specific customer journey",
                        "Order fulfillment experience",
                    ],
                    recommendation=(
                        f"Review customer experience and "
                        f"delivery performance for "
                        f"{lowest_channel}."
                    ),
                )
            )

    return insights


def generate_anomaly_insights(
    predictions: pd.DataFrame,
) -> list[Insight]:
    """
    Generate an insight from the anomaly detector.
    """

    insights = []

    total_records = len(predictions)

    anomaly_count = int(
        predictions[
            "is_predicted_anomaly"
        ].sum()
    )

    if total_records == 0:
        return insights

    anomaly_rate = (
        anomaly_count
        / total_records
        * 100
    )

    if anomaly_count > 0:

        severity = (
            "high"
            if anomaly_rate >= 5
            else "medium"
        )

        top_anomalies = (
            predictions
            .loc[
                predictions[
                    "is_predicted_anomaly"
                ] == 1
            ]
            .sort_values(
                "anomaly_score"
            )
            .head(5)
        )

        top_records = []

        for _, row in top_anomalies.iterrows():

            top_records.append(
                {
                    "order_id": str(
                        row["order_id"]
                    ),
                    "anomaly_score": round(
                        float(
                            row["anomaly_score"]
                        ),
                        4,
                    ),
                }
            )

        insights.append(
            create_insight(
                title="Potential anomalies detected",
                finding=(
                    f"The anomaly detector identified "
                    f"{anomaly_count:,} potentially unusual "
                    f"transactions out of "
                    f"{total_records:,} records "
                    f"({anomaly_rate:.2f}%)."
                ),
                evidence=[
                    {
                        "metric": "predicted_anomaly_count",
                        "count": anomaly_count,
                    },
                    {
                        "metric": "predicted_anomaly_rate",
                        "rate_pct": round(
                            anomaly_rate,
                            2,
                        ),
                    },
                    {
                        "metric": "highest_priority_records",
                        "records": top_records,
                    },
                ],
                severity=severity,
                impact=(
                    "Unusual transactions may represent "
                    "financial, operational, or customer "
                    "experience risks requiring investigation."
                ),
                possible_contributing_factors=[
                    "Unusual transaction amounts",
                    "Unusual discounts or costs",
                    "Operational delays",
                    "Unusual return behavior",
                ],
                recommendation=(
                    "Prioritize the highest-scoring "
                    "transactions for investigation and "
                    "review their underlying business "
                    "attributes."
                ),
            )
        )

    return insights


def generate_insights(
    df: pd.DataFrame,
) -> list[Insight]:
    """
    Run the complete evidence-backed insight pipeline.
    """

    # ---------------------------------------------------------
    # Analytics
    # ---------------------------------------------------------

    calculate_kpis(df)

    monthly_metrics = (
    calculate_monthly_trends(df)
)

    patterns = calculate_patterns(df)

    # ---------------------------------------------------------
    # Feature engineering + ML
    # ---------------------------------------------------------

    engineered_df = engineer_features(df)

    detector = AnomalyDetector()

    predictions = detector.fit_predict(
        engineered_df
    )

    # ---------------------------------------------------------
    # Insight generation
    # ---------------------------------------------------------

    insights = []

    insights.extend(
        generate_trend_insights(
            monthly_metrics
        )
    )

    insights.extend(
        generate_segment_insights(
            patterns
        )
    )

    insights.extend(
        generate_anomaly_insights(
            predictions
        )
    )

    return insights


def print_insights(
    insights: list[Insight],
) -> None:
    """
    Print structured insights in a readable format.
    """

    print()
    print("=" * 80)
    print("INSIGHT ENGINE")
    print("=" * 80)

    print(
        f"Insights generated: {len(insights)}"
    )

    print()

    for index, insight in enumerate(
        insights,
        start=1,
    ):

        print(
            f"[{index}] {insight.title}"
        )

        print(
            f"Severity: {insight.severity.upper()}"
        )

        print(
            f"Finding: {insight.finding}"
        )

        print(
            f"Impact: {insight.impact}"
        )

        print(
            "Possible contributing factors:"
        )

        for factor in (
            insight.possible_contributing_factors
        ):
            print(
                f"  - {factor}"
            )

        print(
            f"Recommendation: "
            f"{insight.recommendation}"
        )

        print(
            "Evidence:"
        )

        for evidence_item in insight.evidence:

            print(
                f"  {evidence_item}"
            )

        print("-" * 80)


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

    insights = generate_insights(
        df
    )

    print_insights(
        insights
    )


if __name__ == "__main__":
    main()