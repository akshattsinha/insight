import math
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.analysis.kpis import calculate_kpis
from src.analysis.patterns import calculate_patterns
from src.analysis.trends import calculate_trends
from src.features.engineering import engineer_features
from src.insights.engine import generate_insights
from src.models.anomaly_detector import (
    AnomalyDetector,
    evaluate_predictions,
)


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


def make_json_safe(value):
    """
    Convert pandas/NumPy values into JSON-safe values.

    NaN and infinite values are converted to None.
    Dictionaries and lists are handled recursively.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if hasattr(value, "item"):
        try:
            return make_json_safe(
                value.item()
            )
        except (ValueError, TypeError):
            pass

    return value


@router.post("/analyze")
def analyze_dataset(
    file: UploadFile = File(...),
):
    """
    Analyze an uploaded CSV dataset.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as temp_file:

            temp_path = Path(
                temp_file.name
            )

            contents = file.file.read()

            if not contents:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty.",
                )

            temp_file.write(contents)

        df = pd.read_csv(temp_path)

        insights = generate_insights(df)

        response = {
            "status": "success",
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "insight_count": len(insights),
            "insights": [
                {
                    "title": insight.title,
                    "finding": insight.finding,
                    "evidence": insight.evidence,
                    "severity": insight.severity,
                    "impact": insight.impact,
                    "possible_contributing_factors": (
                        insight.possible_contributing_factors
                    ),
                    "recommendation": (
                        insight.recommendation
                    ),
                }
                for insight in insights
            ],
        }

        return make_json_safe(response)

    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV contains no data.",
        ) from exc

    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid CSV.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset analysis failed: {exc}",
        ) from exc

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()


@router.get("/analyze-synthetic")
def analyze_synthetic_dataset():
    """
    Analyze the built-in synthetic retail dataset.

    Returns the complete analytics payload
    required by the Streamlit dashboard.
    """

    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "synthetic_retail_data.csv"
    )

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Synthetic dataset not found.",
        )

    try:

        # =========================================================
        # LOAD DATASET
        # =========================================================

        df = pd.read_csv(
            dataset_path
        )

        # =========================================================
        # 1. CORE KPIs
        # =========================================================

        kpis = calculate_kpis(df)

        # =========================================================
        # 2. MONTHLY TRENDS
        # =========================================================

        trends = calculate_trends(df)

        trend_records = trends.to_dict(
            orient="records"
        )

        # =========================================================
        # 3. PATTERN ANALYSIS
        # =========================================================

        patterns = calculate_patterns(
            df
        )

        correlation_matrix = patterns[
            "correlation_matrix"
        ].to_dict()

        strong_correlations = patterns[
            "strong_correlations"
        ]

        segment_performance = {
            segment: dataframe.to_dict(
                orient="records"
            )
            for segment, dataframe
            in patterns[
                "segment_performance"
            ].items()
        }

        segment_extremes = patterns[
            "segment_extremes"
        ]

        # =========================================================
        # 4. FEATURE ENGINEERING
        # =========================================================

        engineered_df = engineer_features(
            df
        )

        # =========================================================
        # 5. ANOMALY DETECTION
        # =========================================================

        detector = AnomalyDetector()

        predictions = detector.fit_predict(
            engineered_df
        )

        evaluation = evaluate_predictions(
            predictions
        )

        predicted_anomalies = predictions[
            predictions[
                "is_predicted_anomaly"
            ]
            == 1
        ].copy()

        predicted_anomalies = (
            predicted_anomalies
            .sort_values(
                "anomaly_score",
                ascending=True,
            )
            .head(20)
        )

        anomaly_records = []

        for _, row in (
            predicted_anomalies.iterrows()
        ):

            anomaly_records.append(
                {
                    "order_id": str(
                        row["order_id"]
                    ),
                    "anomaly_score": float(
                        row["anomaly_score"]
                    ),
                    "revenue": float(
                        row["revenue"]
                    ),
                    "profit": float(
                        row["profit"]
                    ),
                    "delivery_days": float(
                        row["delivery_days"]
                    ),
                    "return_status": str(
                        row["return_status"]
                    ),
                    "category": str(
                        row["category"]
                    ),
                    "region": str(
                        row["region"]
                    ),
                }
            )

        predicted_count = int(
            predictions[
                "is_predicted_anomaly"
            ].sum()
        )

        anomaly_rate = (
            predicted_count
            / len(predictions)
            * 100
        )

        # =========================================================
        # 6. INSIGHT ENGINE
        # =========================================================

        insights = generate_insights(
            df
        )

        insight_records = [
            {
                "title": insight.title,
                "finding": insight.finding,
                "evidence": insight.evidence,
                "severity": insight.severity,
                "impact": insight.impact,
                "possible_contributing_factors": (
                    insight.possible_contributing_factors
                ),
                "recommendation": (
                    insight.recommendation
                ),
            }
            for insight in insights
        ]

        # =========================================================
        # 7. COMPLETE DASHBOARD PAYLOAD
        # =========================================================

        response = {
            "status": "success",

            "dataset": {
                "filename": (
                    dataset_path.name
                ),
                "rows": len(df),
                "columns": len(df.columns),
            },

            "kpis": kpis,

            "trends": trend_records,

            "segments": (
                segment_performance
            ),

            "segment_extremes": (
                segment_extremes
            ),

            "correlations": {
                "strong": (
                    strong_correlations
                ),
                "matrix": (
                    correlation_matrix
                ),
            },

            "anomalies": {
                "predicted_count": (
                    predicted_count
                ),
                "predicted_rate": round(
                    anomaly_rate,
                    2,
                ),

                "evaluation": {
                    "precision": float(
                        evaluation[
                            "precision"
                        ]
                    ),
                    "recall": float(
                        evaluation[
                            "recall"
                        ]
                    ),
                    "f1_score": float(
                        evaluation[
                            "f1_score"
                        ]
                    ),
                    "confusion_matrix": (
                        evaluation[
                            "confusion_matrix"
                        ].tolist()
                    ),
                },

                "records": (
                    anomaly_records
                ),
            },

            "insight_count": len(
                insights
            ),

            "insights": (
                insight_records
            ),
        }

        return make_json_safe(
            response
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Synthetic dataset "
                "analysis failed: "
                f"{exc}"
            ),
        ) from exc