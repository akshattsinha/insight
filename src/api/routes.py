import tempfile
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.insights.engine import generate_insights
from src.analysis.kpis import calculate_kpis


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
):
    """
    Analyze an uploaded CSV dataset and return
    verified business insights.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A file must be provided.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            delete=False,
        ) as temp_file:
            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        df = pd.read_csv(temp_path)

        insights = generate_insights(df)

        return {
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
                    "recommendation": insight.recommendation,
                }
                for insight in insights
            ],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
@router.get("/analyze-synthetic")
def analyze_synthetic_dataset():
    """
    Analyze the built-in INSIGHT synthetic retail dataset.
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
        df = pd.read_csv(dataset_path)

        kpis = calculate_kpis(df)
        insights = generate_insights(df)

        return {
            "status": "success",
            "filename": dataset_path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "kpis": kpis,
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
                    "recommendation": insight.recommendation,
                }
                for insight in insights
            ],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Synthetic dataset analysis failed: {exc}",
        ) from exc