from pathlib import Path
import sys
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile


# -------------------------------------------------------------------
# Project path setup
# -------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# -------------------------------------------------------------------
# Analysis imports
# -------------------------------------------------------------------

from analysis.kpis import calculate_kpis
from analysis.patterns import calculate_patterns
from analysis.profiler import generate_profile
from analysis.trends import calculate_trends
from analysis.validator import validate_dataset
from features.engineering import engineer_features
from models.anomaly_detector import (
    AnomalyDetector,
    evaluate_predictions,
)
from insights.engine import generate_insights

from chatbot.service import InsightChatbot


# -------------------------------------------------------------------
# Router
# -------------------------------------------------------------------

router = APIRouter()


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

SYNTHETIC_DATASET_PATH = (
    DATA_DIR / "synthetic_retail_data.csv"
)


# -------------------------------------------------------------------
# Chatbot
# -------------------------------------------------------------------

chatbot = InsightChatbot()


# -------------------------------------------------------------------
# Serialization helpers
# -------------------------------------------------------------------

def _serialize_value(value: Any) -> Any:
    """
    Convert pandas, NumPy, and other scalar values
    into JSON-safe values.
    """

    # NumPy arrays
    if hasattr(value, "tolist"):
        try:
            return _serialize_object(
                value.tolist()
            )
        except Exception:
            pass

    # NumPy scalar values
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    # Pandas timestamps
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    # None
    if value is None:
        return None

    # Pandas missing scalar values
    try:
        if (
            pd.api.types.is_scalar(value)
            and pd.isna(value)
        ):
            return None
    except (TypeError, ValueError):
        pass

    return value


def _serialize_dataframe(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame into JSON-safe records.
    """

    records = df.to_dict(
        orient="records"
    )

    serialized = []

    for record in records:

        serialized_record = {
            key: _serialize_value(value)
            for key, value in record.items()
        }

        serialized.append(
            serialized_record
        )

    return serialized


def _serialize_object(
    value: Any,
) -> Any:
    """
    Recursively convert analysis objects into
    JSON-compatible structures.
    """

    if isinstance(value, pd.DataFrame):
        return _serialize_dataframe(value)

    if isinstance(value, pd.Series):
        return [
            _serialize_value(item)
            for item in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            str(key): _serialize_object(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _serialize_object(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _serialize_object(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        return {
            key: _serialize_object(item)
            for key, item in value.__dict__.items()
        }

    return _serialize_value(value)


# -------------------------------------------------------------------
# Ground-truth helper
# -------------------------------------------------------------------

GROUND_TRUTH_COLUMNS = {
    "is_injected_anomaly",
    "anomaly_type",
}


def _has_ground_truth(
    df: pd.DataFrame,
) -> bool:
    """
    Determine whether the dataset contains the
    synthetic evaluation ground-truth columns.
    """

    return GROUND_TRUTH_COLUMNS.issubset(
        set(df.columns)
    )


# -------------------------------------------------------------------
# Core analysis pipeline
# -------------------------------------------------------------------

def run_analysis_pipeline(
    df: pd.DataFrame,
    source_name: str = "dataset",
) -> dict[str, Any]:
    """
    Run the complete INSIGHT analysis pipeline.

    Pipeline:

        Raw Data
            ↓
        Validation
            ↓
        Profiling
            ↓
        KPIs
            ↓
        Trends
            ↓
        Patterns
            ↓
        Feature Engineering
            ↓
        Isolation Forest
            ↓
        Evaluation (when ground truth exists)
            ↓
        Insights
    """

    if df.empty:
        raise ValueError(
            "The supplied dataset is empty."
        )

    # ---------------------------------------------------------------
    # 1. Determine whether evaluation ground truth exists
    # ---------------------------------------------------------------

    has_ground_truth = _has_ground_truth(df)

    # ---------------------------------------------------------------
    # 2. Validation
    # ---------------------------------------------------------------

    validation = validate_dataset(df)

    if not validation.get("valid", False):
        return {
            "success": False,
            "source": source_name,
            "error": "Dataset validation failed.",
            "validation": validation,
        }

    # ---------------------------------------------------------------
    # 3. Dataset profile
    # ---------------------------------------------------------------

    profile = generate_profile(df)

    # ---------------------------------------------------------------
    # 4. KPI calculations
    # ---------------------------------------------------------------

    kpis = calculate_kpis(df)

    # ---------------------------------------------------------------
    # 5. Trend analysis
    # ---------------------------------------------------------------

    trends = calculate_trends(df)

    # ---------------------------------------------------------------
    # 6. Pattern analysis
    # ---------------------------------------------------------------

    patterns = calculate_patterns(df)

    # ---------------------------------------------------------------
    # 7. Feature engineering + ML
    #
    # IMPORTANT:
    # Ground-truth columns are not used as ML features.
    # ---------------------------------------------------------------

    engineered_df = engineer_features(df)

    detector = AnomalyDetector(
        contamination=0.03,
        random_state=42,
    )

    predictions = detector.fit_predict(
        engineered_df
    )

    # ---------------------------------------------------------------
    # 8. Evaluation
    #
    # Evaluation only makes sense when ground truth exists.
    # ---------------------------------------------------------------

    evaluation = None

    if has_ground_truth:
        evaluation = evaluate_predictions(
            predictions
        )

    # ---------------------------------------------------------------
    # 9. Insight generation
    #
    # Reuse already-computed analytics instead of recalculating
    # trends, patterns, and anomaly detection.
    # ---------------------------------------------------------------

    insights = generate_insights(
        df,
        monthly_metrics=trends,
        patterns=patterns,
        predictions=predictions,
    )

    # ---------------------------------------------------------------
    # 10. Build response
    # ---------------------------------------------------------------

    result = {
        "success": True,
        "source": source_name,

        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(df.columns),
        },

        "validation": validation,

        "profile": profile,

        "kpis": kpis,

        "trends": trends,

        "patterns": patterns,

        "features": {
            "rows": int(len(engineered_df)),
            "columns": int(
                len(engineered_df.columns)
            ),
            "column_names": list(
                engineered_df.columns
            ),
        },

        "anomalies": {
            "predictions": predictions,
            "evaluation": evaluation,
        },

        "insights": insights,
    }

    return _serialize_object(result)


# -------------------------------------------------------------------
# API status
# -------------------------------------------------------------------

@router.get("/api/status")
def api_status() -> dict[str, Any]:
    """
    Return API status information.
    """

    return {
        "status": "online",
        "service": "INSIGHT API",
    }


# -------------------------------------------------------------------
# Synthetic dataset analysis
# -------------------------------------------------------------------

@router.get("/api/analyze-synthetic")
def analyze_synthetic() -> dict[str, Any]:
    """
    Analyze the bundled synthetic retail dataset.
    """

    if not SYNTHETIC_DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Synthetic dataset not found: "
                f"{SYNTHETIC_DATASET_PATH}"
            ),
        )

    try:
        df = pd.read_csv(
            SYNTHETIC_DATASET_PATH
        )

        return run_analysis_pipeline(
            df,
            source_name="synthetic",
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc


# -------------------------------------------------------------------
# Uploaded CSV analysis
# -------------------------------------------------------------------

@router.post("/api/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Analyze an uploaded CSV dataset.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        from io import BytesIO

        df = pd.read_csv(
            BytesIO(contents)
        )

        return run_analysis_pipeline(
            df,
            source_name=file.filename,
        )

    except HTTPException:
        raise

    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV contains no data.",
        ) from exc

    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not parse the uploaded CSV: "
                f"{exc}"
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc


# -------------------------------------------------------------------
# AI Assistant
# -------------------------------------------------------------------

@router.post("/api/chat")
async def chat(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Ask the INSIGHT AI Assistant a question.

    The optional `analysis` field allows the chatbot
    to reason over the dataset that has already been
    analyzed instead of silently switching to another
    dataset.
    """

    question = payload.get("question")

    if not isinstance(question, str):
        raise HTTPException(
            status_code=400,
            detail="Question must be a string.",
        )

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    analysis = payload.get("analysis")

    try:
        response = chatbot.ask(
            question,
            analysis=analysis,
        )

        return _serialize_object(response)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot failed: {exc}",
        ) from exc