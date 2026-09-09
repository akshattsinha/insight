from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "INSIGHT_API_URL",
    "http://insight:8080",
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="INSIGHT",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK / BLACK ENTERPRISE THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background-color: #050505;
        color: #ffffff;
    }

    .main {
        background-color: #050505;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ========================================================
       TEXT
    ======================================================== */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #ffffff !important;
    }

    p {
        color: #d1d5db;
    }

    label {
        color: #d1d5db !important;
    }

    /* ========================================================
       SIDEBAR
    ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #222222;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #d1d5db !important;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
    }

    /* ========================================================
       METRICS
    ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #111111;
        border: 1px solid #292929;
        border-radius: 12px;
        padding: 18px;
    }

    div[data-testid="stMetric"] label {
        color: #9ca3af !important;
    }

    div[data-testid="stMetric"] div {
        color: #ffffff !important;
    }

    /* ========================================================
       CONTAINERS
    ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0d0d0d;
        border-color: #292929 !important;
        border-radius: 12px;
    }

    /* ========================================================
       SELECT BOX
    ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #111111;
        border-color: #333333;
        color: #ffffff;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    /* ========================================================
       TEXT AREA
    ======================================================== */

    textarea {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    textarea::placeholder {
        color: #6b7280 !important;
    }

    /* ========================================================
       BUTTON
    ======================================================== */

    button {
        border-radius: 8px !important;
    }

    /* ========================================================
       DATAFRAME
    ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #292929;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ========================================================
       ALERTS
    ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ========================================================
       DIVIDERS
    ======================================================== */

    hr {
        border-color: #252525 !important;
    }

    /* ========================================================
       CODE
    ======================================================== */

    pre {
        background-color: #0b0b0b !important;
        border: 1px solid #252525;
        border-radius: 10px;
    }

    /* ========================================================
       SCROLLBAR
    ======================================================== */

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #050505;
    }

    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555555;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD ANALYSIS FROM API
# ============================================================

@st.cache_data(ttl=60)
def load_analysis() -> dict[str, Any]:

    response = requests.get(
        f"{API_URL}/api/analyze-synthetic",
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# FORMATTERS
# ============================================================

def format_currency(value: Any) -> str:

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0

    if abs(value) >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.1f}K"

    return f"₹{value:,.0f}"


def format_number(value: Any) -> str:

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def format_percentage(value: Any) -> str:

    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


# ============================================================
# LOAD DATA
# ============================================================

try:

    analysis = load_analysis()

except requests.exceptions.RequestException as exc:

    st.error(
        "Unable to connect to the INSIGHT backend."
    )

    st.code(
        f"Backend: {API_URL}\n\n"
        f"Error: {exc}"
    )

    st.stop()

except Exception as exc:

    st.error(
        "Unable to load INSIGHT analytics."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# EXTRACT DATA
# ============================================================

kpis = analysis.get(
    "kpis",
    {},
)

trends = analysis.get(
    "trends",
    [],
)

segments = analysis.get(
    "segments",
    {},
)

segment_extremes = analysis.get(
    "segment_extremes",
    [],
)

correlations = analysis.get(
    "correlations",
    {},
)

anomalies = analysis.get(
    "anomalies",
    {},
)

insights = analysis.get(
    "insights",
    [],
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("INSIGHT")

    st.caption(
        "AI-Powered Decision Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Analytics",
            "Segments",
            "Investigation Center",
            "Risk & Anomalies",
            "AI Assistant",
        ],
    )

    st.divider()

    st.subheader(
        "System Status"
    )

    st.success(
        "Analytics Engine"
    )

    st.success(
        "Anomaly Detection"
    )

    st.info(
        "Local LLM: Ollama"
    )

    st.divider()

    st.subheader(
        "Dataset"
    )

    st.write(
        "Synthetic Retail Dataset"
    )

    st.write(
        "10,000 transactions"
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.caption(
        "EXECUTIVE OVERVIEW"
    )

    st.title(
        "Business Performance"
    )

    st.write(
        "A verified view of revenue, profitability, "
        "customer behavior, operations, and emerging risks."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI ROW 1
    # --------------------------------------------------------

    columns = st.columns(4)

    with columns[0]:

        st.metric(
            "Revenue",
            format_currency(
                kpis.get(
                    "total_revenue",
                    0,
                )
            ),
        )

    with columns[1]:

        st.metric(
            "Profit",
            format_currency(
                kpis.get(
                    "total_profit",
                    0,
                )
            ),
        )

    with columns[2]:

        st.metric(
            "Profit Margin",
            format_percentage(
                kpis.get(
                    "profit_margin",
                    0,
                )
            ),
        )

    with columns[3]:

        st.metric(
            "Return Rate",
            format_percentage(
                kpis.get(
                    "return_rate",
                    0,
                )
            ),
        )

    # --------------------------------------------------------
    # KPI ROW 2
    # --------------------------------------------------------

    columns = st.columns(4)

    with columns[0]:

        st.metric(
            "Orders",
            format_number(
                kpis.get(
                    "total_orders",
                    0,
                )
            ),
        )

    with columns[1]:

        st.metric(
            "Customers",
            format_number(
                kpis.get(
                    "total_customers",
                    0,
                )
            ),
        )

    with columns[2]:

        st.metric(
            "Average Order Value",
            format_currency(
                kpis.get(
                    "average_order_value",
                    0,
                )
            ),
        )

    with columns[3]:

        st.metric(
            "Average Delivery",
            f'{kpis.get("average_delivery_time", 0):.2f} days',
        )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.divider()

    st.header(
        "Performance Trajectory"
    )

    st.write(
        "Monthly revenue and profit performance."
    )

    trend_df = pd.DataFrame(
        trends
    )

    if not trend_df.empty:

        trend_df["month"] = pd.to_datetime(
            trend_df["month"]
        )

        trend_df = trend_df.sort_values(
            "month"
        )

        chart_df = trend_df.set_index(
            "month"
        )[
            [
                "revenue",
                "profit",
            ]
        ]

        st.line_chart(
            chart_df,
            height=400,
        )

    # --------------------------------------------------------
    # PRIORITY SIGNALS
    # --------------------------------------------------------

    st.divider()

    st.header(
        "Priority Signals"
    )

    st.write(
        "Highest-priority findings identified by the analytics engine."
    )

    if insights:

        count = min(
            3,
            len(insights),
        )

        columns = st.columns(
            count
        )

        for index in range(count):

            insight = insights[index]

            with columns[index]:

                with st.container(
                    border=True
                ):

                    st.caption(
                        f"SIGNAL {index + 1:02d}"
                    )

                    st.subheader(
                        str(
                            insight.get(
                                "title",
                                "Finding",
                            )
                        )
                    )

                    st.write(
                        str(
                            insight.get(
                                "finding",
                                "",
                            )
                        )
                    )

                    st.caption(
                        "Severity: "
                        + str(
                            insight.get(
                                "severity",
                                "LOW",
                            )
                        ).upper()
                    )

    # --------------------------------------------------------
    # OPERATING SNAPSHOT
    # --------------------------------------------------------

    st.divider()

    st.header(
        "Operating Snapshot"
    )

    columns = st.columns(3)

    with columns[0]:

        st.metric(
            "Customer Satisfaction",
            f'{kpis.get("average_satisfaction", 0):.2f}/5',
        )

    with columns[1]:

        st.metric(
            "Predicted Anomalies",
            format_number(
                anomalies.get(
                    "predicted_count",
                    0,
                )
            ),
        )

    with columns[2]:

        st.metric(
            "Anomaly Rate",
            format_percentage(
                anomalies.get(
                    "predicted_rate",
                    0,
                )
            ),
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.caption(
        "ANALYTICS"
    )

    st.title(
        "Performance Analytics"
    )

    st.write(
        "Explore verified trends, profitability, returns, "
        "and statistical relationships."
    )

    trend_df = pd.DataFrame(
        trends
    )

    if trend_df.empty:

        st.warning(
            "No trend data available."
        )

    else:

        trend_df["month"] = pd.to_datetime(
            trend_df["month"]
        )

        trend_df = trend_df.sort_values(
            "month"
        )

        st.header(
            "Revenue"
        )

        st.line_chart(
            trend_df.set_index("month")[
                ["revenue"]
            ],
            height=320,
        )

        st.header(
            "Profit"
        )

        st.line_chart(
            trend_df.set_index("month")[
                ["profit"]
            ],
            height=320,
        )

        st.header(
            "Return Rate"
        )

        st.line_chart(
            trend_df.set_index("month")[
                ["return_rate"]
            ],
            height=300,
        )

        st.header(
            "Profit Margin"
        )

        st.line_chart(
            trend_df.set_index("month")[
                ["profit_margin"]
            ],
            height=300,
        )

        st.divider()

        st.header(
            "Monthly Performance"
        )

        display_df = trend_df.copy()

        display_df["month"] = (
            display_df["month"]
            .dt.strftime("%Y-%m")
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------------

    st.divider()

    st.header(
        "Strong Statistical Relationships"
    )

    st.write(
        "These relationships represent correlations, "
        "not confirmed causal relationships."
    )

    strong_correlations = correlations.get(
        "strong",
        [],
    )

    if strong_correlations:

        correlation_df = pd.DataFrame(
            strong_correlations
        )

        st.dataframe(
            correlation_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No strong correlations identified."
        )


# ============================================================
# SEGMENTS
# ============================================================

elif page == "Segments":

    st.caption(
        "SEGMENT INTELLIGENCE"
    )

    st.title(
        "Understand Performance by Segment"
    )

    st.write(
        "Compare business performance across categories, "
        "regions, channels, customer segments, and payment methods."
    )

    dimensions = list(
        segments.keys()
    )

    if not dimensions:

        st.warning(
            "No segment data available."
        )

    else:

        selected_dimension = st.selectbox(
            "Business Dimension",
            dimensions,
        )

        segment_data = segments.get(
            selected_dimension,
            [],
        )

        if segment_data:

            segment_df = pd.DataFrame(
                segment_data
            )

            numeric_columns = [
                column
                for column in segment_df.columns
                if pd.api.types.is_numeric_dtype(
                    segment_df[column]
                )
            ]

            preferred_metrics = [
                "revenue",
                "profit",
                "return_rate",
                "average_delivery_days",
                "average_satisfaction",
            ]

            default_metric = next(
                (
                    metric
                    for metric in preferred_metrics
                    if metric in numeric_columns
                ),
                (
                    numeric_columns[0]
                    if numeric_columns
                    else None
                ),
            )

            if default_metric:

                selected_metric = st.selectbox(
                    "Metric",
                    numeric_columns,
                    index=numeric_columns.index(
                        default_metric
                    ),
                )

                st.header(
                    "Segment Comparison"
                )

                chart_df = segment_df.copy()

                if (
                    selected_dimension
                    in chart_df.columns
                ):

                    chart_df = chart_df.set_index(
                        selected_dimension
                    )

                    st.bar_chart(
                        chart_df[
                            [selected_metric]
                        ],
                        height=400,
                    )

                st.header(
                    "Segment Performance"
                )

                st.dataframe(
                    segment_df,
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        st.header(
            "Segment Extremes"
        )

        if segment_extremes:

            extremes_df = pd.DataFrame(
                segment_extremes
            )

            st.dataframe(
                extremes_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No segment extremes available."
            )


# ============================================================
# INVESTIGATION CENTER
# ============================================================

elif page == "Investigation Center":

    st.caption(
        "INVESTIGATION CENTER"
    )

    st.title(
        "From Signal to Decision"
    )

    st.write(
        "Inspect verified findings, supporting evidence, "
        "business impact, hypotheses, and recommended actions."
    )

    if not insights:

        st.warning(
            "No insights were generated."
        )

    else:

        # ----------------------------------------------------
        # FINDING SELECTOR
        # ----------------------------------------------------

        st.header(
            "Detected Findings"
        )

        insight_options = []

        for index, insight in enumerate(
            insights
        ):

            title = str(
                insight.get(
                    "title",
                    "Untitled finding",
                )
            )

            severity = str(
                insight.get(
                    "severity",
                    "low",
                )
            ).upper()

            insight_options.append(
                f"{index + 1:02d} | "
                f"{severity} | "
                f"{title}"
            )

        selected_label = st.selectbox(
            "Select a finding to investigate",
            insight_options,
        )

        selected_index = (
            insight_options.index(
                selected_label
            )
        )

        selected_insight = insights[
            selected_index
        ]

        st.divider()

        # ----------------------------------------------------
        # INVESTIGATION
        # ----------------------------------------------------

        st.caption(
            f"INVESTIGATION {selected_index + 1:02d}"
        )

        st.header(
            str(
                selected_insight.get(
                    "title",
                    "Untitled finding",
                )
            )
        )

        severity = str(
            selected_insight.get(
                "severity",
                "LOW",
            )
        ).upper()

        if severity == "HIGH":

            st.error(
                "Severity: HIGH"
            )

        elif severity == "MEDIUM":

            st.warning(
                "Severity: MEDIUM"
            )

        else:

            st.success(
                f"Severity: {severity}"
            )

        # ----------------------------------------------------
        # FINDING
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Finding"
        )

        finding = selected_insight.get(
            "finding",
            "No finding available.",
        )

        st.text(
            str(finding)
        )

        # ----------------------------------------------------
        # VERIFIED EVIDENCE
        # ----------------------------------------------------

        st.subheader(
            "Verified Evidence"
        )

        evidence = selected_insight.get(
            "evidence",
            {},
        )

        if isinstance(
            evidence,
            dict,
        ):

            evidence_rows = []

            for key, value in evidence.items():

                evidence_rows.append(
                    {
                        "Evidence": str(key),
                        "Value": str(value),
                    }
                )

            if evidence_rows:

                evidence_df = pd.DataFrame(
                    evidence_rows
                )

                st.dataframe(
                    evidence_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.text(
                    "No evidence available."
                )

        elif isinstance(
            evidence,
            list,
        ):

            evidence_rows = []

            for index, item in enumerate(
                evidence
            ):

                if isinstance(
                    item,
                    dict,
                ):

                    for key, value in item.items():

                        evidence_rows.append(
                            {
                                "Evidence": str(key),
                                "Value": str(value),
                            }
                        )

                else:

                    evidence_rows.append(
                        {
                            "Evidence": str(
                                index + 1
                            ),
                            "Value": str(item),
                        }
                    )

            if evidence_rows:

                evidence_df = pd.DataFrame(
                    evidence_rows
                )

                st.dataframe(
                    evidence_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.text(
                    "No evidence available."
                )

        else:

            st.text(
                str(evidence)
            )

        # ----------------------------------------------------
        # BUSINESS IMPACT
        # ----------------------------------------------------

        st.subheader(
            "Business Impact"
        )

        impact = selected_insight.get(
            "impact",
            "No business impact information available.",
        )

        st.text(
            str(impact)
        )

        # ----------------------------------------------------
        # POSSIBLE CONTRIBUTING FACTORS
        # ----------------------------------------------------

        st.subheader(
            "Possible Contributing Factors"
        )

        factors = selected_insight.get(
            "possible_contributing_factors",
            [],
        )

        if isinstance(
            factors,
            list,
        ):

            for factor in factors:

                st.text(
                    "• " + str(factor)
                )

        elif factors:

            st.text(
                str(factors)
            )

        else:

            st.text(
                "No possible contributing factors identified."
            )

        st.caption(
            "Possible factors are hypotheses for investigation "
            "and should not be interpreted as confirmed causes."
        )

        # ----------------------------------------------------
        # RECOMMENDED ACTION
        # ----------------------------------------------------

        st.subheader(
            "Recommended Action"
        )

        recommendation = selected_insight.get(
            "recommendation",
            "No recommendation available.",
        )

        st.info(
            str(recommendation)
        )


# ============================================================
# RISK & ANOMALIES
# ============================================================

elif page == "Risk & Anomalies":

    st.caption(
        "RISK & ANOMALIES"
    )

    st.title(
        "Operational Risk Detection"
    )

    st.write(
        "Isolation Forest identifies unusual transactions "
        "independently of the ground-truth evaluation labels."
    )

    evaluation = anomalies.get(
        "evaluation",
        {},
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    columns = st.columns(5)

    with columns[0]:

        st.metric(
            "Predicted Anomalies",
            format_number(
                anomalies.get(
                    "predicted_count",
                    0,
                )
            ),
        )

    with columns[1]:

        st.metric(
            "Anomaly Rate",
            format_percentage(
                anomalies.get(
                    "predicted_rate",
                    0,
                )
            ),
        )

    with columns[2]:

        precision = (
            float(
                evaluation.get(
                    "precision",
                    0,
                )
            )
            * 100
        )

        st.metric(
            "Precision",
            format_percentage(
                precision
            ),
        )

    with columns[3]:

        recall = (
            float(
                evaluation.get(
                    "recall",
                    0,
                )
            )
            * 100
        )

        st.metric(
            "Recall",
            format_percentage(
                recall
            ),
        )

    with columns[4]:

        f1 = (
            float(
                evaluation.get(
                    "f1_score",
                    0,
                )
            )
            * 100
        )

        st.metric(
            "F1 Score",
            format_percentage(
                f1
            ),
        )

    # --------------------------------------------------------
    # DETECTION PERFORMANCE
    # --------------------------------------------------------

    st.divider()

    st.header(
        "Detection Performance"
    )

    st.write(
        "Evaluation metrics compare Isolation Forest "
        "predictions against the synthetic ground-truth "
        "labels after detection. Ground-truth labels are "
        "not used as model features."
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    confusion_matrix = evaluation.get(
        "confusion_matrix",
        [],
    )

    if confusion_matrix:

        st.subheader(
            "Confusion Matrix"
        )

        confusion_df = pd.DataFrame(
            confusion_matrix,
            index=[
                "Actual Normal",
                "Actual Anomaly",
            ],
            columns=[
                "Predicted Normal",
                "Predicted Anomaly",
            ],
        )

        st.dataframe(
            confusion_df,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # ANOMALY RECORDS
    # --------------------------------------------------------

    st.divider()

    st.header(
        "Highest-Risk Records"
    )

    anomaly_records = anomalies.get(
        "records",
        [],
    )

    if anomaly_records:

        anomaly_df = pd.DataFrame(
            anomaly_records
        )

        st.dataframe(
            anomaly_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No anomaly records returned."
        )


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "AI Assistant":

    st.caption(
        "AI ASSISTANT"
    )

    st.title(
        "Ask INSIGHT"
    )

    st.write(
        "Ask business questions and receive evidence-backed "
        "explanations from the analytics engine."
    )

    st.divider()

    st.subheader(
        "Example Questions"
    )

    example_questions = [
        "Why are returns higher in Fashion?",
        "Which region has the slowest delivery?",
        "Which channel has the lowest satisfaction?",
        "What are the most important anomalies?",
        "What should management investigate first?",
    ]

    for question in example_questions:

        st.write(
            "• " + question
        )

    st.divider()

    question = st.text_area(
        "Ask a business question",
        placeholder=(
            "Example: Why is the return rate higher in Fashion?"
        ),
        height=120,
    )

    st.button(
        "Analyze Question",
        type="primary",
        disabled=True,
    )

    st.caption(
        "AI reasoning will be connected to the analytics "
        "engine and local Ollama model in the next step."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "INSIGHT · Compute first. Explain second. "
    "· Decision Intelligence Platform · v0.1.0"
)