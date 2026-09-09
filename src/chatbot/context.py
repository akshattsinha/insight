from __future__ import annotations

from typing import Any

import pandas as pd

from src.chatbot.router import QuestionRoute


class AnalyticsContextBuilder:
    """
    Builds verified analytical context for the LLM.

    The preferred path is to consume an analysis result that has
    already been produced by the main analytics pipeline.

    The synthetic dataset remains as a fallback for standalone
    chatbot usage and backwards compatibility.
    """

    def __init__(
        self,
        data_path: str = "data/synthetic_retail_data.csv",
    ):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Load the fallback synthetic dataset."""
        return pd.read_csv(self.data_path)

    @staticmethod
    def _clean_value(value: Any) -> Any:
        """Convert pandas/numpy values into JSON-safe Python values."""

        if isinstance(value, dict):
            return {
                str(key): AnalyticsContextBuilder._clean_value(val)
                for key, val in value.items()
            }

        if isinstance(value, list):
            return [
                AnalyticsContextBuilder._clean_value(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                AnalyticsContextBuilder._clean_value(item)
                for item in value
            ]

        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass

        if not isinstance(value, (dict, list, tuple)):
            try:
                if pd.isna(value):
                    return None
            except (ValueError, TypeError):
                pass

        return value

    def _build_returns_context(
        self,
        patterns: dict[str, Any],
        kpis: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "question_scope": "returns",
            "kpis": self._clean_value(kpis),
            "return_patterns": self._clean_value(
                patterns.get("returns", patterns)
            ),
        }

    def _build_delivery_context(
        self,
        patterns: dict[str, Any],
        kpis: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "question_scope": "delivery",
            "kpis": self._clean_value(kpis),
            "delivery_patterns": self._clean_value(
                patterns.get("delivery", patterns)
            ),
        }

    def _build_satisfaction_context(
        self,
        patterns: dict[str, Any],
        kpis: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "question_scope": "satisfaction",
            "kpis": self._clean_value(kpis),
            "satisfaction_patterns": self._clean_value(
                patterns.get("satisfaction", patterns)
            ),
        }

    def _build_revenue_context(
        self,
        patterns: dict[str, Any],
        trends: Any,
        kpis: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "question_scope": "revenue",
            "kpis": self._clean_value(kpis),
            "trends": self._clean_value(trends),
            "revenue_patterns": self._clean_value(
                patterns.get("revenue", patterns)
            ),
        }

    def _build_profit_context(
        self,
        patterns: dict[str, Any],
        trends: Any,
        kpis: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "question_scope": "profit",
            "kpis": self._clean_value(kpis),
            "trends": self._clean_value(trends),
            "profit_patterns": self._clean_value(
                patterns.get("profit", patterns)
            ),
        }

    def _build_segment_context(
        self,
        patterns: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "question_scope": "segments",
            "segment_patterns": self._clean_value(
                patterns.get("segments", patterns)
            ),
        }

    def build_context_from_analysis(
        self,
        route: QuestionRoute,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build targeted AI context from an already-computed INSIGHT
        analysis result.

        This prevents the chatbot from independently loading and
        re-analyzing the synthetic dataset when the dashboard/API
        already has verified analytical results.
        """

        kpis = analysis.get("kpis", {})
        patterns = analysis.get("patterns", {})
        trends = analysis.get("trends", {})
        anomalies = analysis.get("anomalies", {})
        dataset = analysis.get("dataset", {})

        if route.category == "returns":

            context = self._build_returns_context(
                patterns,
                kpis,
            )

        elif route.category == "delivery":

            context = self._build_delivery_context(
                patterns,
                kpis,
            )

        elif route.category == "satisfaction":

            context = self._build_satisfaction_context(
                patterns,
                kpis,
            )

        elif route.category == "revenue":

            context = self._build_revenue_context(
                patterns,
                trends,
                kpis,
            )

        elif route.category == "profit":

            context = self._build_profit_context(
                patterns,
                trends,
                kpis,
            )

        elif route.category == "segments":

            context = self._build_segment_context(
                patterns,
            )

        elif route.category == "anomalies":

            evaluation = anomalies.get("evaluation")

            prediction_records = anomalies.get(
                "predictions",
                [],
            )

            predicted_count = None

            if isinstance(evaluation, dict):

                predicted_count = evaluation.get(
                    "predicted_anomalies"
                )

            if (
                predicted_count is None
                and isinstance(prediction_records, list)
            ):

                predicted_count = sum(
                    1
                    for record in prediction_records
                    if (
                        isinstance(record, dict)
                        and record.get(
                            "is_predicted_anomaly"
                        ) == 1
                    )
                )

            context = {
                "question_scope": "anomaly detection",
                "predicted_count": predicted_count,
                "evaluation": self._clean_value(
                    evaluation
                ),
                "records": self._clean_value(
                    prediction_records
                ),
            }

        else:

            context = {
                "question_scope": "general business overview",
                "kpis": self._clean_value(kpis),
                "trends": self._clean_value(trends),
                "patterns": self._clean_value(patterns),
            }

        return {
            "dataset": {
                "rows": dataset.get("rows"),
                "columns": dataset.get("columns"),
                "source": dataset.get("source"),
                "filename": dataset.get("filename"),
            },
            **context,
        }

    def build_context(
        self,
        route: QuestionRoute,
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build targeted analytical context according to the question
        category.

        If an existing verified analysis is supplied, reuse it.

        If no analysis is supplied, preserve the existing fallback
        behavior by loading the synthetic dataset.
        """

        if analysis is not None:

            return self.build_context_from_analysis(
                route,
                analysis,
            )

        data = self.load_data()

        context: dict[str, Any] = {
            "dataset": {
                "rows": int(len(data)),
                "columns": int(len(data.columns)),
                "source": "synthetic",
                "filename": self.data_path,
            }
        }

        if route.category == "returns":

            return {
                **context,
                "question_scope": "returns",
                "return_rate": self._clean_value(
                    data["return_status"]
                    .eq("Returned")
                    .mean()
                ),
                "return_amount": self._clean_value(
                    data["return_amount"].sum()
                ),
                "return_by_category": self._clean_value(
                    data.groupby("category")["return_status"]
                    .apply(
                        lambda x: x.eq("Returned").mean()
                    )
                    .sort_values(
                        ascending=False
                    )
                    .to_dict()
                ),
            }

        if route.category == "delivery":

            return {
                **context,
                "question_scope": "delivery",
                "average_delivery_days": self._clean_value(
                    data["delivery_days"].mean()
                ),
                "delivery_by_region": self._clean_value(
                    data.groupby("region")["delivery_days"]
                    .mean()
                    .sort_values(
                        ascending=False
                    )
                    .to_dict()
                ),
            }

        if route.category == "satisfaction":

            return {
                **context,
                "question_scope": "satisfaction",
                "average_satisfaction": self._clean_value(
                    data["satisfaction_score"].mean()
                ),
                "satisfaction_by_channel": self._clean_value(
                    data.groupby("channel")[
                        "satisfaction_score"
                    ]
                    .mean()
                    .sort_values()
                    .to_dict()
                ),
            }

        if route.category == "revenue":

            return {
                **context,
                "question_scope": "revenue",
                "total_revenue": self._clean_value(
                    data["revenue"].sum()
                ),
                "average_revenue": self._clean_value(
                    data["revenue"].mean()
                ),
                "revenue_by_category": self._clean_value(
                    data.groupby("category")["revenue"]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .to_dict()
                ),
            }

        if route.category == "profit":

            return {
                **context,
                "question_scope": "profit",
                "total_profit": self._clean_value(
                    data["profit"].sum()
                ),
                "average_profit": self._clean_value(
                    data["profit"].mean()
                ),
                "profit_by_category": self._clean_value(
                    data.groupby("category")["profit"]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .to_dict()
                ),
            }

        if route.category == "segments":

            return {
                **context,
                "question_scope": "segments",
                "segment_revenue": self._clean_value(
                    data.groupby("customer_segment")[
                        "revenue"
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .to_dict()
                ),
                "segment_profit": self._clean_value(
                    data.groupby("customer_segment")[
                        "profit"
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .to_dict()
                ),
            }

        if route.category == "anomalies":

            return {
                **context,
                "question_scope": "anomalies",
                "ground_truth_available": (
                    "is_injected_anomaly"
                    in data.columns
                ),
                "ground_truth_anomalies": (
                    self._clean_value(
                        data["is_injected_anomaly"].sum()
                    )
                    if "is_injected_anomaly"
                    in data.columns
                    else None
                ),
            }

        return {
            **context,
            "question_scope": "general business overview",
            "total_revenue": self._clean_value(
                data["revenue"].sum()
            ),
            "total_profit": self._clean_value(
                data["profit"].sum()
            ),
            "average_satisfaction": self._clean_value(
                data["satisfaction_score"].mean()
            ),
            "average_delivery_days": self._clean_value(
                data["delivery_days"].mean()
            ),
        }