from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="INSIGHT — Decision Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONFIGURATION
# =============================================================================

API_URL = os.getenv(
    "INSIGHT_API_URL",
    "http://insight:8080",
)


# =============================================================================
# DARK / BLACK THEME
# =============================================================================

st.markdown(
    """
    <style>

    /* ---------------------------------------------------------
       GLOBAL
    --------------------------------------------------------- */

    .stApp {
        background-color: #050505;
        color: #f5f5f5;
    }

    .main {
        background-color: #050505;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #050505;
    }

    [data-testid="stHeader"] {
        background-color: #050505;
    }

    [data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid #1f1f1f;
    }

    [data-testid="stSidebar"] * {
        color: #f5f5f5;
    }


    /* ---------------------------------------------------------
       TYPOGRAPHY
    --------------------------------------------------------- */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #ffffff !important;
    }

    p,
    span,
    label,
    div {
        color: inherit;
    }


    /* ---------------------------------------------------------
       METRIC CARDS
    --------------------------------------------------------- */

    [data-testid="stMetric"] {
        background-color: #0d0d0d;
        border: 1px solid #242424;
        border-radius: 12px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #8d8d8d !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    [data-testid="stMetricDelta"] {
        color: #aaaaaa !important;
    }


    /* ---------------------------------------------------------
       BUTTONS
    --------------------------------------------------------- */

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #303030;
        background-color: #111111;
        color: #ffffff;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #ffffff;
        color: #ffffff;
        background-color: #191919;
    }


    /* ---------------------------------------------------------
       TEXT INPUTS
    --------------------------------------------------------- */

    textarea,
    input {
        background-color: #101010 !important;
        color: #ffffff !important;
        border: 1px solid #303030 !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #666666 !important;
    }


    /* ---------------------------------------------------------
       DATAFRAMES
    --------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #242424;
        border-radius: 10px;
    }


    /* ---------------------------------------------------------
       ALERTS
    --------------------------------------------------------- */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* ---------------------------------------------------------
       DIVIDERS
    --------------------------------------------------------- */

    hr {
        border-color: #222222;
    }


    /* ---------------------------------------------------------
       CUSTOM TEXT
    --------------------------------------------------------- */

    .insight-brand {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #ffffff;
        margin-bottom: 0;
    }

    .insight-subtitle {
        color: #777777;
        font-size: 0.9rem;
        margin-top: -5px;
    }

    .section-label {
        color: #777777;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .status-online {
        color: #63d471;
        font-weight: 700;
    }

    .status-offline {
        color: #ff6464;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# API HELPERS
# =============================================================================


def check_api() -> bool:
    """
    Check whether the INSIGHT FastAPI backend is reachable.
    """

    try:
        response = requests.get(
            f"{API_URL.rstrip('/')}/health",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


@st.cache_data(ttl=60)
def load_analysis() -> dict[str, Any]:
    """
    Load the built-in synthetic dataset analysis from FastAPI.

    The dashboard does not perform the analytics itself.
    FastAPI runs the analytics pipeline and returns verified
    analytical results.
    """

    response = requests.get(
        f"{API_URL.rstrip('/')}/api/analyze-synthetic",
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


def ask_ai_assistant(question: str) -> dict[str, Any]:
    """
    Send a natural-language business question to the
    INSIGHT chatbot endpoint.
    """

    endpoint = (
        f"{API_URL.rstrip('/')}/api/chat"
    )

    try:

        response = requests.post(
            endpoint,
            json={
                "question": question,
            },
            timeout=180,
        )

        if response.status_code != 200:

            try:

                error_data = response.json()

                error_message = error_data.get(
                    "detail",
                    error_data.get(
                        "message",
                        "Unknown API error.",
                    ),
                )

            except ValueError:

                error_message = (
                    response.text
                    or "Unknown API error."
                )

            return {
                "status": "error",
                "message": str(
                    error_message
                ),
            }

        data = response.json()

        if data.get("status") != "success":

            return {
                "status": "error",
                "message": data.get(
                    "message",
                    "AI Assistant returned an error.",
                ),
            }

        return data

    except requests.exceptions.Timeout:

        return {
            "status": "error",
            "message": (
                "The AI Assistant timed out. "
                "Ollama may still be processing the request."
            ),
        }

    except requests.exceptions.ConnectionError:

        return {
            "status": "error",
            "message": (
                "Could not connect to the INSIGHT API. "
                "Make sure the backend container is running."
            ),
        }

    except requests.exceptions.RequestException as error:

        return {
            "status": "error",
            "message": (
                f"API request failed: {error}"
            ),
        }

    except ValueError:

        return {
            "status": "error",
            "message": (
                "The API returned an invalid response."
            ),
        }

    except Exception as error:

        return {
            "status": "error",
            "message": (
                f"Unexpected error: {error}"
            ),
        }


# =============================================================================
# DATA HELPERS
# =============================================================================


def get_nested(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve nested dictionary values.
    """

    current: Any = data

    for key in keys:

        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def format_currency(value: Any) -> str:
    """
    Format a numeric value as Indian Rupees.
    """

    try:

        value = float(value)

        if abs(value) >= 1_00_00_000:

            return f"₹{value / 1_00_00_000:.2f} Cr"

        if abs(value) >= 1_00_000:

            return f"₹{value / 1_00_000:.2f} L"

        if abs(value) >= 1_000:

            return f"₹{value:,.0f}"

        return f"₹{value:,.2f}"

    except (TypeError, ValueError):

        return "N/A"


def format_number(value: Any) -> str:
    """
    Format a numeric value.
    """

    try:

        value = float(value)

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    except (TypeError, ValueError):

        return "N/A"


def format_percent(value: Any) -> str:
    """
    Format a percentage.
    """

    try:

        return f"{float(value):.2f}%"

    except (TypeError, ValueError):

        return "N/A"


def dataframe_from_value(
    value: Any,
) -> pd.DataFrame:
    """
    Convert common API structures into a DataFrame.
    """

    if isinstance(value, list):

        if not value:
            return pd.DataFrame()

        return pd.DataFrame(value)

    if isinstance(value, dict):

        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame()

    return pd.DataFrame()


def normalize_segment_data(
    segment_data: Any,
) -> pd.DataFrame:
    """
    Convert segment performance API data into
    a DataFrame.

    The API may return a dictionary keyed by segment.
    """

    if not isinstance(segment_data, dict):
        return pd.DataFrame()

    rows = []

    for segment_name, values in segment_data.items():

        if not isinstance(values, list):
            continue

        for row in values:

            if not isinstance(row, dict):
                continue

            item = dict(row)

            item["segment_dimension"] = segment_name

            rows.append(item)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =============================================================================
# SESSION STATE
# =============================================================================


if "active_page" not in st.session_state:

    st.session_state.active_page = "Overview"


if "ai_question" not in st.session_state:

    st.session_state.ai_question = ""


if "ai_messages" not in st.session_state:

    st.session_state.ai_messages = []


if "selected_investigation" not in st.session_state:

    st.session_state.selected_investigation = 0


# =============================================================================
# LOAD API DATA
# =============================================================================


api_online = check_api()


if not api_online:

    st.error(
        "INSIGHT API is offline. "
        "Start the FastAPI container before using the dashboard."
    )

    st.code(
        "docker compose up -d --build insight dashboard",
        language="bash",
    )

    st.stop()


try:

    analysis = load_analysis()

except requests.RequestException as error:

    st.error(
        "Unable to load the INSIGHT analytical pipeline."
    )

    st.code(
        str(error)
    )

    st.stop()

except Exception as error:

    st.error(
        "Unexpected dashboard error."
    )

    st.code(
        str(error)
    )

    st.stop()


# =============================================================================
# EXTRACT API RESULTS
# =============================================================================


kpis = analysis.get(
    "kpis",
    {},
)

trends = analysis.get(
    "trends",
    {},
)

patterns = analysis.get(
    "patterns",
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

validation = analysis.get(
    "validation",
    {},
)


# =============================================================================
# SIDEBAR
# =============================================================================


with st.sidebar:

    st.markdown(
        '<div class="insight-brand">INSIGHT</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="insight-subtitle">'
        "AI-Powered Decision Intelligence"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="section-label">Navigation</div>',
        unsafe_allow_html=True,
    )

    pages = [
        "Overview",
        "Analytics",
        "Segments",
        "Investigation Center",
        "Incident Center",
        "Risk & Anomalies",
        "AI Assistant",
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(
            st.session_state.active_page
        ),
        label_visibility="collapsed",
    )

    st.session_state.active_page = selected_page

    st.divider()

    st.markdown(
        '<div class="section-label">System Status</div>',
        unsafe_allow_html=True,
    )

    st.success("● Analytics Engine Online")

    st.success("● Insight Engine Online")

    st.success("● API Online")

    st.caption(
        f"Backend: {API_URL}"
    )

    st.divider()

    st.caption(
        "INSIGHT computes evidence first, "
        "then uses GenAI for explanation."
    )


# =============================================================================
# HEADER
# =============================================================================


st.markdown(
    '<div class="insight-brand">INSIGHT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="insight-subtitle">'
    "AI-Powered Decision Intelligence Platform"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# =============================================================================
# OVERVIEW
# =============================================================================


# =============================================================================
# INCIDENT INVESTIGATION HELPERS
# =============================================================================


def _safe_float(value: Any) -> float | None:
    """Convert a scalar value to float without raising dashboard errors."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any, default: str = "N/A") -> str:
    """Return a clean display string for a scalar value."""
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


def build_incident_drivers(
    incident_row: pd.Series,
    all_predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Find the strongest statistically unusual engineered signals.

    This is descriptive evidence only. It does not establish fraud, intent,
    causation, policy violation, or a confirmed business loss.
    """
    definitions = [
        ("profit_margin", "Profit Margin", "Profit relative to revenue."),
        ("cost_to_revenue_ratio", "Cost / Revenue", "Operational cost relative to revenue."),
        ("shipping_to_revenue_ratio", "Shipping / Revenue", "Shipping cost relative to revenue."),
        ("discount_ratio", "Discount Ratio", "Discount impact relative to transaction value."),
        ("delivery_delay_score", "Delivery Delay", "Engineered delivery-delay signal."),
        ("satisfaction_risk", "Satisfaction Risk", "Engineered customer-satisfaction risk signal."),
        ("return_ratio", "Return Ratio", "Returned amount relative to revenue."),
        ("transaction_value", "Transaction Value", "Engineered transaction-value signal."),
    ]

    drivers: list[dict[str, Any]] = []
    if all_predictions.empty:
        return drivers

    for column, name, description in definitions:
        if column not in all_predictions.columns or column not in incident_row.index:
            continue

        value = _safe_float(incident_row.get(column))
        if value is None:
            continue

        series = pd.to_numeric(all_predictions[column], errors="coerce").dropna()
        if series.empty:
            continue

        percentile = float((series <= value).mean() * 100.0)
        extremeness = abs(percentile - 50.0)
        if extremeness < 20.0:
            continue

        if percentile >= 97.5:
            direction, label = "Very High", "Extreme"
        elif percentile >= 75:
            direction, label = "High", "Unusual"
        elif percentile <= 2.5:
            direction, label = "Very Low", "Extreme"
        else:
            direction, label = "Low", "Unusual"

        drivers.append({
            "name": name,
            "column": column,
            "description": description,
            "value": value,
            "percentile": percentile,
            "extremeness": extremeness,
            "direction": direction,
            "label": label,
        })

    drivers.sort(key=lambda item: item["extremeness"], reverse=True)
    return drivers[:6]


def calculate_incident_metrics(row: pd.Series) -> dict[str, float | None]:
    """Calculate descriptive business ratios from original transaction fields."""
    revenue = _safe_float(row.get("revenue"))
    profit = _safe_float(row.get("profit"))
    shipping = _safe_float(row.get("shipping_cost"))
    operational = _safe_float(row.get("operational_cost"))
    return_amount = _safe_float(row.get("return_amount"))

    return {
        "profit_margin": (profit / revenue * 100.0) if revenue not in (None, 0) and profit is not None else None,
        "cost_ratio": ((shipping or 0) + (operational or 0)) / revenue * 100.0
        if revenue not in (None, 0) and (shipping is not None or operational is not None) else None,
        "return_ratio": return_amount / revenue * 100.0
        if revenue not in (None, 0) and return_amount is not None else None,
    }


def find_peer_transactions(
    incident_row: pd.Series,
    all_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Find same-context transactions for a transparent peer comparison."""
    if all_predictions.empty:
        return pd.DataFrame()

    peer = all_predictions.copy()

    # Prefer the strongest business context available.
    context_columns = ["category", "region", "channel", "customer_segment"]
    for column in context_columns:
        value = incident_row.get(column)
        if column in peer.columns and value is not None:
            try:
                if not pd.isna(value):
                    peer = peer[peer[column].astype(str) == str(value)]
            except (TypeError, ValueError):
                pass

    if len(peer) < 10:
        # Fall back to category + region when the full context is too narrow.
        peer = all_predictions.copy()
        for column in ["category", "region"]:
            value = incident_row.get(column)
            if column in peer.columns and value is not None:
                try:
                    if not pd.isna(value):
                        peer = peer[peer[column].astype(str) == str(value)]
                except (TypeError, ValueError):
                    pass

    order_id = incident_row.get("order_id")
    if order_id is not None and "order_id" in peer.columns:
        peer = peer[peer["order_id"].astype(str) != str(order_id)]

    return peer


def build_peer_comparison(
    incident_row: pd.Series,
    peers: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build incident-vs-peer medians using original business fields."""
    comparisons = [
        ("Revenue", "revenue", "currency"),
        ("Profit", "profit", "currency"),
        ("Delivery Days", "delivery_days", "number"),
        ("Satisfaction", "satisfaction_score", "number"),
        ("Discount", "discount_pct", "percent"),
        ("Return Amount", "return_amount", "currency"),
    ]

    results = []
    if peers.empty:
        return results

    for label, column, kind in comparisons:
        if column not in peers.columns:
            continue
        incident_value = _safe_float(incident_row.get(column))
        peer_values = pd.to_numeric(peers[column], errors="coerce").dropna()
        if incident_value is None or peer_values.empty:
            continue

        median = float(peer_values.median())
        difference = incident_value - median
        if median != 0:
            difference_pct = difference / abs(median) * 100.0
        else:
            difference_pct = None

        if kind == "currency":
            incident_display = format_currency(incident_value)
            peer_display = format_currency(median)
        elif kind == "percent":
            incident_display = format_percent(incident_value)
            peer_display = format_percent(median)
        else:
            incident_display = f"{incident_value:.2f}"
            peer_display = f"{median:.2f}"

        results.append({
            "label": label,
            "incident": incident_display,
            "peer": peer_display,
            "difference_pct": difference_pct,
            "peer_count": len(peer_values),
        })

    return results


def find_similar_incidents(
    incident_row: pd.Series,
    all_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return nearby flagged incidents using transparent business context."""
    if all_predictions.empty or "is_predicted_anomaly" not in all_predictions.columns:
        return pd.DataFrame()

    flagged = all_predictions.copy()
    flagged["is_predicted_anomaly"] = flagged["is_predicted_anomaly"].astype(bool)
    flagged = flagged[flagged["is_predicted_anomaly"]].copy()

    if flagged.empty:
        return flagged

    order_id = str(incident_row.get("order_id", ""))
    if "order_id" in flagged.columns:
        flagged = flagged[flagged["order_id"].astype(str) != order_id]

    # Score contextual similarity. This is a deterministic ranking, not an ML claim.
    score = pd.Series(0.0, index=flagged.index)
    for column, weight in [
        ("category", 4.0),
        ("region", 3.0),
        ("channel", 2.0),
        ("customer_segment", 1.0),
    ]:
        if column not in flagged.columns:
            continue
        target = _safe_text(incident_row.get(column), "")
        score += (flagged[column].astype(str) == target).astype(float) * weight

    if "anomaly_score" in flagged.columns:
        flagged["_abs_score"] = pd.to_numeric(flagged["anomaly_score"], errors="coerce").abs()
        flagged = flagged.assign(_context_score=score)
        flagged = flagged.sort_values(
            ["_context_score", "_abs_score"],
            ascending=[False, False],
        )
    else:
        flagged = flagged.assign(_context_score=score)
        flagged = flagged.sort_values("_context_score", ascending=False)

    return flagged.head(5)


def build_ai_investigation_prompt(
    incident_row: pd.Series,
    drivers: list[dict[str, Any]],
    peer_comparisons: list[dict[str, Any]],
) -> str:
    """Create a grounded incident-investigation question for the chatbot."""
    facts = []
    for field in [
        "order_id", "order_date", "region", "channel", "customer_segment",
        "category", "product", "quantity", "revenue", "return_status",
        "return_amount", "delivery_days", "satisfaction_score", "shipping_cost",
        "operational_cost", "profit", "payment_method", "anomaly_score",
    ]:
        if field in incident_row.index:
            value = incident_row[field]
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            facts.append(f"{field}={value}")

    driver_text = "; ".join(
        f"{d['name']} {d['direction']} at {d['percentile']:.1f}th percentile"
        for d in drivers
    ) or "No individual engineered signal crossed the investigation threshold."

    peer_text = "; ".join(
        f"{p['label']}: incident {p['incident']} vs peer median {p['peer']}"
        for p in peer_comparisons
    ) or "No peer comparison was available."

    return (
        "Investigate this INSIGHT incident using ONLY the verified facts below. "
        "Explain what is unusual, compare it with peers, identify plausible "
        "investigation hypotheses, and recommend concrete verification steps. "
        "Do not claim fraud, intent, causation, or confirmed loss unless the "
        "provided evidence explicitly proves it. Clearly distinguish facts from hypotheses.\n\n"
        f"VERIFIED TRANSACTION FACTS: {'; '.join(facts)}\n"
        f"UNUSUAL ENGINEERED SIGNALS: {driver_text}\n"
        f"PEER COMPARISONS: {peer_text}"
    )


if selected_page == "Overview":

    st.header("Decision Overview")

    st.write(
        "A high-level view of business performance, "
        "trends, risks, and evidence-backed findings."
    )

    st.divider()

    # -------------------------------------------------------------------------
    # KPI CARDS
    # -------------------------------------------------------------------------

    total_revenue = kpis.get(
        "total_revenue",
        0,
    )

    total_orders = kpis.get(
        "total_orders",
        kpis.get(
            "orders",
            0,
        ),
    )

    total_customers = kpis.get(
        "total_customers",
        kpis.get(
            "customers",
            0,
        ),
    )

    total_profit = kpis.get(
        "total_profit",
        kpis.get(
            "profit",
            0,
        ),
    )

    profit_margin = kpis.get(
        "profit_margin",
        0,
    )

    return_rate = kpis.get(
        "return_rate",
        0,
    )

    average_order_value = kpis.get(
        "average_order_value",
        kpis.get(
            "aov",
            0,
        ),
    )

    average_delivery = kpis.get(
        "average_delivery_days",
        0,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Revenue",
            format_currency(
                total_revenue
            ),
        )

    with col2:

        st.metric(
            "Orders",
            format_number(
                total_orders
            ),
        )

    with col3:

        st.metric(
            "Customers",
            format_number(
                total_customers
            ),
        )

    with col4:

        st.metric(
            "Profit",
            format_currency(
                total_profit
            ),
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:

        st.metric(
            "Profit Margin",
            format_percent(
                profit_margin
            ),
        )

    with col6:

        st.metric(
            "Return Rate",
            format_percent(
                return_rate
            ),
        )

    with col7:

        st.metric(
            "Average Order Value",
            format_currency(
                average_order_value
            ),
        )

    with col8:

        st.metric(
            "Avg Delivery",
            f"{float(average_delivery):.2f} days"
            if average_delivery is not None
            else "N/A",
        )

    st.divider()

    # -------------------------------------------------------------------------
    # INSIGHT SUMMARY
    # -------------------------------------------------------------------------

    st.subheader(
        "Priority Findings"
    )

    if not insights:

        st.info(
            "No evidence-backed insights were generated."
        )

    else:

        for index, insight in enumerate(
            insights[:4],
            start=1,
        ):

            if not isinstance(insight, dict):
                continue

            title = insight.get(
                "title",
                f"Insight {index}",
            )

            finding = insight.get(
                "finding",
                "No finding available.",
            )

            severity = str(
                insight.get(
                    "severity",
                    "MEDIUM",
                )
            ).upper()

            impact = insight.get(
                "impact",
                "",
            )

            st.markdown(
                f"### {index}. {title}"
            )

            st.write(
                finding
            )

            if severity:

                st.caption(
                    f"Severity: {severity}"
                )

            if impact:

                st.caption(
                    f"Impact: {impact}"
                )

            investigate_col, _ = st.columns([1, 4])

            with investigate_col:

                if st.button(
                    "Investigate →",
                    key=f"investigate_overview_{index}",
                    use_container_width=True,
                ):
                    st.session_state.selected_investigation = index - 1
                    st.session_state.active_page = "Investigation Center"
                    st.rerun()

            if index < min(
                4,
                len(insights),
            ):

                st.divider()

    st.divider()

    # -------------------------------------------------------------------------
    # DATASET STATUS
    # -------------------------------------------------------------------------

    st.subheader(
        "Dataset Health"
    )

    validation_columns = st.columns(4)

    with validation_columns[0]:

        st.metric(
            "Rows",
            format_number(
                validation.get(
                    "rows",
                    10_000,
                )
            ),
        )

    with validation_columns[1]:

        st.metric(
            "Columns",
            format_number(
                validation.get(
                    "columns",
                    22,
                )
            ),
        )

    with validation_columns[2]:

        st.metric(
            "Missing Values",
            format_number(
                validation.get(
                    "missing_values",
                    0,
                )
            ),
        )

    with validation_columns[3]:

        st.metric(
            "Duplicate Orders",
            format_number(
                validation.get(
                    "duplicate_order_ids",
                    0,
                )
            ),
        )


# =============================================================================
# ANALYTICS
# =============================================================================


elif selected_page == "Analytics":

    st.header("Analytics")

    st.write(
        "Computed trends, relationships, and business performance "
        "from the validated dataset."
    )

    st.divider()

    # -------------------------------------------------------------------------
    # MONTHLY TRENDS
    # -------------------------------------------------------------------------

    st.subheader(
        "Revenue & Profit Trend"
    )

    # calculate_trends() returns the monthly trend records directly.
    # Keep compatibility with dictionary-based responses as well.
    if isinstance(trends, dict):
        monthly_data = trends.get(
            "monthly",
            trends.get(
                "data",
                [],
            ),
        )
    elif isinstance(trends, list):
        monthly_data = trends
    else:
        monthly_data = []

    monthly_df = dataframe_from_value(
        monthly_data
    )

    if not monthly_df.empty:

        if "month" in monthly_df.columns:

            monthly_df["month"] = (
                monthly_df["month"]
                .astype(str)
            )

            monthly_df = monthly_df.set_index(
                "month"
            )

        chart_columns = [
            column
            for column in [
                "revenue",
                "profit",
            ]
            if column in monthly_df.columns
        ]

        if chart_columns:

            st.line_chart(
                monthly_df[
                    chart_columns
                ]
            )

        st.subheader(
            "Monthly Performance"
        )

        display_columns = [
            column
            for column in [
                "month",
                "revenue",
                "profit",
                "orders",
                "customers",
                "returns",
                "return_rate",
                "average_delivery_days",
                "average_satisfaction",
            ]
            if column in monthly_df.reset_index().columns
        ]

        display_df = monthly_df.reset_index()

        if display_columns:

            st.dataframe(
                display_df[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "Monthly trend data is not available."
        )

    st.divider()

    # -------------------------------------------------------------------------
    # EXTREME MONTHS
    # -------------------------------------------------------------------------

    st.subheader(
        "Performance Extremes"
    )

    # The current calculate_trends() response is a list of monthly
    # records, so derive performance extremes from the same verified
    # monthly data already loaded above.
    extreme_months = {}

    if not monthly_df.empty:
        extreme_metric_columns = [
            column
            for column in [
                "revenue",
                "profit",
                "orders",
                "customers",
                "returns",
                "average_delivery_days",
                "average_satisfaction",
                "return_rate",
                "profit_margin",
                "average_order_value",
            ]
            if column in monthly_df.columns
        ]

        for metric in extreme_metric_columns:
            metric_series = pd.to_numeric(
                monthly_df[metric],
                errors="coerce",
            ).dropna()

            if metric_series.empty:
                continue

            highest_index = metric_series.idxmax()
            lowest_index = metric_series.idxmin()

            extreme_months[metric] = {
                "highest": {
                    "month": str(highest_index),
                    "value": metric_series.loc[highest_index],
                },
                "lowest": {
                    "month": str(lowest_index),
                    "value": metric_series.loc[lowest_index],
                },
            }

    if isinstance(extreme_months, dict):

        extreme_rows = []

        for metric, values in extreme_months.items():

            if not isinstance(values, dict):
                continue

            highest = values.get(
                "highest",
                {},
            )

            lowest = values.get(
                "lowest",
                {},
            )

            extreme_rows.append(
                {
                    "Metric": metric.replace(
                        "_",
                        " ",
                    ).title(),
                    "Highest Month": highest.get(
                        "month",
                        "N/A",
                    ),
                    "Highest Value": highest.get(
                        "value",
                        "N/A",
                    ),
                    "Lowest Month": lowest.get(
                        "month",
                        "N/A",
                    ),
                    "Lowest Value": lowest.get(
                        "value",
                        "N/A",
                    ),
                }
            )

        if extreme_rows:

            st.dataframe(
                pd.DataFrame(
                    extreme_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # -------------------------------------------------------------------------
    # CORRELATIONS
    # -------------------------------------------------------------------------

    st.subheader(
        "Strong Relationships"
    )

    st.caption(
        "Correlation indicates statistical association; "
        "it does not establish causation."
    )

    correlations = patterns.get(
        "strong_correlations",
        [],
    )

    correlation_df = dataframe_from_value(
        correlations
    )

    if not correlation_df.empty:

        st.dataframe(
            correlation_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No strong correlations were identified."
        )


# =============================================================================
# SEGMENTS
# =============================================================================


elif selected_page == "Segments":

    st.header("Segment Intelligence")

    st.write(
        "Compare business performance across categories, "
        "regions, channels, customer segments, and payment methods."
    )

    st.divider()

    segment_performance = patterns.get(
        "segment_performance",
        {},
    )

    if not isinstance(
        segment_performance,
        dict,
    ):

        segment_performance = {}

    available_segments = list(
        segment_performance.keys()
    )

    if not available_segments:

        st.info(
            "Segment performance data is not available."
        )

    else:

        selected_segment = st.selectbox(
            "Select dimension",
            available_segments,
        )

        selected_data = segment_performance.get(
            selected_segment,
            [],
        )

        segment_df = dataframe_from_value(
            selected_data
        )

        if not segment_df.empty:

            st.subheader(
                selected_segment.replace(
                    "_",
                    " ",
                ).title()
            )

            # -------------------------------------------------------------
            # BEST / WORST
            # -------------------------------------------------------------

            metric_col1, metric_col2 = st.columns(2)

            if (
                "revenue" in segment_df.columns
                and not segment_df.empty
            ):

                best_revenue = segment_df.loc[
                    segment_df["revenue"].idxmax()
                ]

                with metric_col1:

                    st.metric(
                        "Highest Revenue Segment",
                        str(
                            best_revenue[
                                selected_segment
                            ]
                        ),
                        format_currency(
                            best_revenue[
                                "revenue"
                            ]
                        ),
                    )

            if (
                "profit" in segment_df.columns
                and not segment_df.empty
            ):

                best_profit = segment_df.loc[
                    segment_df["profit"].idxmax()
                ]

                with metric_col2:

                    st.metric(
                        "Highest Profit Segment",
                        str(
                            best_profit[
                                selected_segment
                            ]
                        ),
                        format_currency(
                            best_profit[
                                "profit"
                            ]
                        ),
                    )

            st.divider()

            # -------------------------------------------------------------
            # SEGMENT CHART
            # -------------------------------------------------------------

            chart_metric = st.selectbox(
                "Compare metric",
                [
                    column
                    for column in [
                        "revenue",
                        "profit",
                        "orders",
                        "returns",
                        "return_rate",
                        "average_delivery_days",
                        "average_satisfaction",
                        "profit_margin",
                        "average_order_value",
                    ]
                    if column in segment_df.columns
                ],
            )

            chart_df = segment_df[
                [
                    selected_segment,
                    chart_metric,
                ]
            ].copy()

            chart_df = chart_df.set_index(
                selected_segment
            )

            st.bar_chart(
                chart_df
            )

            st.divider()

            # -------------------------------------------------------------
            # TABLE
            # -------------------------------------------------------------

            st.subheader(
                "Detailed Segment Performance"
            )

            st.dataframe(
                segment_df,
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# INVESTIGATION CENTER
# =============================================================================


elif selected_page == "Investigation Center":

    st.header("Investigation Center")

    st.write(
        "Review evidence-backed findings and understand "
        "what requires investigation or action."
    )

    st.caption(
        "Finding → Verified Evidence → Possible Factors → Recommended Action"
    )

    st.divider()

    if not insights:

        st.info(
            "No insights are currently available."
        )

    else:

        insight_titles = []

        valid_insights = []

        for index, insight in enumerate(
            insights
        ):

            if not isinstance(
                insight,
                dict,
            ):
                continue

            title = insight.get(
                "title",
                f"Insight {index + 1}",
            )

            insight_titles.append(
                f"{index + 1}. {title}"
            )

            valid_insights.append(
                insight
            )

        if valid_insights:

            default_investigation = min(
                max(
                    int(st.session_state.get("selected_investigation", 0)),
                    0,
                ),
                len(valid_insights) - 1,
            )

            selected_index = st.selectbox(
                "Select finding",
                range(
                    len(valid_insights)
                ),
                index=default_investigation,
                format_func=lambda index: insight_titles[
                    index
                ],
            )

            st.session_state.selected_investigation = selected_index

            selected_insight = valid_insights[
                selected_index
            ]

            title = selected_insight.get(
                "title",
                "Untitled Insight",
            )

            finding = selected_insight.get(
                "finding",
                "No finding available.",
            )

            severity = str(
                selected_insight.get(
                    "severity",
                    "MEDIUM",
                )
            ).upper()

            impact = selected_insight.get(
                "impact",
                "No impact information available.",
            )

            evidence = selected_insight.get(
                "evidence",
                [],
            )

            factors = selected_insight.get(
                "possible_contributing_factors",
                [],
            )

            recommendation = selected_insight.get(
                "recommendation",
                "No recommendation available.",
            )

            st.subheader(
                title
            )

            st.write(
                finding
            )

            severity_col, impact_col = st.columns(2)

            with severity_col:

                st.metric(
                    "Severity",
                    severity,
                )

            with impact_col:

                st.write(
                    "**Business Impact**"
                )

                st.write(
                    impact
                )

            st.divider()

            st.subheader(
                "Investigation Brief"
            )

            brief_col1, brief_col2 = st.columns(2)

            with brief_col1:
                st.write("**Observed Finding**")
                st.write(finding)

            with brief_col2:
                st.write("**Decision Priority**")
                st.write(
                    f"{severity} — review the verified evidence before taking action."
                )

            st.divider()

            st.subheader(
                "Verified Evidence"
            )

            if isinstance(
                evidence,
                list,
            ) and evidence:

                evidence_rows = []

                for item in evidence:

                    if isinstance(
                        item,
                        dict,
                    ):

                        for key, value in item.items():

                            evidence_rows.append(
                                {
                                    "Metric": str(
                                        key
                                    ),
                                    "Value": str(
                                        value
                                    ),
                                }
                            )

                    else:

                        evidence_rows.append(
                            {
                                "Evidence": str(
                                    item
                                ),
                            }
                        )

                if evidence_rows:

                    st.dataframe(
                        pd.DataFrame(
                            evidence_rows
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            else:

                st.info(
                    "No structured evidence was returned."
                )

            st.divider()

            st.caption(
                "Evidence boundary: observed relationships are not treated as causal explanations unless the analytics engine explicitly establishes causality."
            )

            st.subheader(
                "Possible Contributing Factors"
            )

            if isinstance(
                factors,
                list,
            ) and factors:

                for factor in factors:

                    st.write(
                        f"• {factor}"
                    )

            else:

                st.write(
                    "The available evidence does not establish a specific cause."
                )

            st.divider()

            st.subheader(
                "Recommended Action"
            )

            st.info(
                recommendation
            )


# =============================================================================
# INCIDENT CENTER
# =============================================================================


elif selected_page == "Incident Center":

    st.header("Incident Center")
    st.write(
        "Investigate statistically unusual transactions with a structured, "
        "evidence-first workflow — from detection to context, peer comparison, "
        "and next investigation action."
    )
    st.divider()

    incident_records = anomalies.get("predictions", []) if isinstance(anomalies, dict) else []
    incident_df = dataframe_from_value(incident_records)

    if not incident_df.empty and "is_predicted_anomaly" in incident_df.columns:
        incident_df["is_predicted_anomaly"] = incident_df["is_predicted_anomaly"].astype(bool)
        incident_df = incident_df[incident_df["is_predicted_anomaly"]].copy()

    if incident_df.empty:
        st.success("No predicted anomalies are currently available.")
    else:
        if "anomaly_score" in incident_df.columns:
            incident_df["anomaly_score"] = pd.to_numeric(
                incident_df["anomaly_score"], errors="coerce"
            )
            incident_df = incident_df.sort_values("anomaly_score", ascending=True)

        incident_count = len(incident_df)
        total_records = validation.get("rows", 0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Open Incidents", format_number(incident_count))
        with c2:
            st.metric("Analyzed Records", format_number(total_records))
        with c3:
            rate = incident_count / total_records * 100 if total_records else 0
            st.metric("Detection Rate", format_percent(rate))
        with c4:
            high_count = 0
            if "anomaly_score" in incident_df.columns:
                high_count = int((incident_df["anomaly_score"] <= -0.15).sum())
            st.metric("High Priority", format_number(high_count))

        st.divider()

        selected_incident = st.session_state.get("selected_incident")

        if selected_incident is not None and selected_incident in incident_df.index:
            row = incident_df.loc[selected_incident]
            order_id = str(row.get("order_id", selected_incident))
            score_value = _safe_float(row.get("anomaly_score"))
            severity = "HIGH" if score_value is not None and score_value <= -0.15 else "MEDIUM"

            if st.button("← Back to Incident Queue", key="back_to_incident_queue"):
                st.session_state.selected_incident = None
                st.rerun()

            # =============================================================
            # INCIDENT HEADER
            # =============================================================
            st.markdown(
                f"""
                <div style="padding:20px;border:1px solid #262626;border-radius:14px;
                            background:#0b0b0b;margin-bottom:18px;">
                    <div style="font-size:12px;letter-spacing:2px;opacity:.6;">INCIDENT INVESTIGATION</div>
                    <div style="font-size:28px;font-weight:700;margin-top:6px;">{order_id}</div>
                    <div style="opacity:.7;margin-top:6px;">
                        Isolation Forest flagged this transaction as statistically unusual.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.metric("Severity", severity)
            with h2:
                st.metric("Anomaly Score", f"{score_value:.4f}" if score_value is not None else "N/A")
            with h3:
                st.metric("Status", "OPEN")
            with h4:
                st.metric("Investigation ID", order_id)

            # =============================================================
            # EXECUTIVE INVESTIGATION SUMMARY
            # =============================================================
            st.divider()
            st.subheader("Investigation Summary")

            metrics = calculate_incident_metrics(row)
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.markdown("**Transaction Context**")
                st.write(f"Category: **{_safe_text(row.get('category'))}**")
                st.write(f"Region: **{_safe_text(row.get('region'))}**")
                st.write(f"Channel: **{_safe_text(row.get('channel'))}**")
                st.write(f"Segment: **{_safe_text(row.get('customer_segment'))}**")
            with summary_cols[1]:
                st.markdown("**Financial Profile**")
                st.write(f"Revenue: **{format_currency(row.get('revenue', 0))}**")
                st.write(f"Profit: **{format_currency(row.get('profit', 0))}**")
                if metrics["profit_margin"] is not None:
                    st.write(f"Profit margin: **{metrics['profit_margin']:.2f}%**")
                else:
                    st.write("Profit margin: **N/A**")
            with summary_cols[2]:
                st.markdown("**Operational Profile**")
                st.write(f"Delivery: **{_safe_text(row.get('delivery_days'))} days**")
                st.write(f"Satisfaction: **{_safe_text(row.get('satisfaction_score'))}**")
                st.write(f"Return status: **{_safe_text(row.get('return_status'))}**")

            # =============================================================
            # WHY FLAGGED — DEEP EVIDENCE
            # =============================================================
            st.divider()
            st.subheader("Why Was This Incident Flagged?")
            st.caption(
                "The model detected a multivariate anomaly. The cards below identify "
                "individual engineered signals that are unusually positioned in the dataset."
            )

            all_prediction_df = dataframe_from_value(incident_records)
            drivers = build_incident_drivers(row, all_prediction_df)

            if drivers:
                for driver in drivers:
                    percentile = driver["percentile"]
                    if percentile >= 97.5 or percentile <= 2.5:
                        emphasis = "EXTREME"
                    else:
                        emphasis = "UNUSUAL"

                    with st.container(border=True):
                        a, b, c = st.columns([3, 1.2, 1.2])
                        with a:
                            st.markdown(f"**{driver['name']}**")
                            st.caption(driver["description"])
                        with b:
                            st.metric("Direction", driver["direction"])
                        with c:
                            st.metric("Signal", emphasis)

                        d1, d2, d3 = st.columns(3)
                        with d1:
                            st.write(f"**Observed:** `{driver['value']:.4f}`")
                        with d2:
                            st.write(f"**Dataset percentile:** `{percentile:.1f}th`")
                        with d3:
                            st.write(f"**Extremeness:** `{driver['extremeness']:.1f}`")
            else:
                st.info(
                    "No individual engineered feature crossed the evidence threshold. "
                    "The anomaly can still result from a combination of multiple moderate signals."
                )

            # =============================================================
            # PEER / BENCHMARK ANALYSIS
            # =============================================================
            st.divider()
            st.subheader("Peer Benchmark")

            peers = find_peer_transactions(row, all_prediction_df)
            peer_count = len(peers)
            st.caption(
                "Comparison uses transactions sharing the incident's business context "
                "where sufficient records are available. Values are computed from the dataset."
            )

            if peer_count:
                st.write(f"**Peer population:** {format_number(peer_count)} comparable transactions")
                peer_comparisons = build_peer_comparison(row, peers)

                if peer_comparisons:
                    peer_cols = st.columns(3)
                    for index, comparison in enumerate(peer_comparisons):
                        with peer_cols[index % 3]:
                            difference = comparison["difference_pct"]
                            if difference is None:
                                delta = "N/A"
                            else:
                                delta = f"{difference:+.1f}% vs peer median"
                            st.metric(
                                comparison["label"],
                                comparison["incident"],
                                delta,
                            )
                            st.caption(f"Peer median: {comparison['peer']}")
                else:
                    st.info("Peer records exist, but the available numeric fields are insufficient for comparison.")
            else:
                peer_comparisons = []
                st.info("No sufficiently comparable peer population was available for this incident.")

            # =============================================================
            # SIMILAR INCIDENTS
            # =============================================================
            st.divider()
            st.subheader("Similar Flagged Incidents")
            st.caption(
                "Prioritized by matching business context such as category, region, "
                "channel, and customer segment. This is a transparent similarity ranking."
            )

            similar = find_similar_incidents(row, all_prediction_df)
            if not similar.empty:
                display_columns = [
                    c for c in [
                        "order_id", "category", "region", "channel", "revenue",
                        "profit", "delivery_days", "anomaly_score",
                    ] if c in similar.columns
                ]
                display_df = similar[display_columns].copy()
                rename_map = {
                    "order_id": "Incident",
                    "category": "Category",
                    "region": "Region",
                    "channel": "Channel",
                    "revenue": "Revenue",
                    "profit": "Profit",
                    "delivery_days": "Delivery Days",
                    "anomaly_score": "Anomaly Score",
                }
                display_df = display_df.rename(columns=rename_map)
                if "Revenue" in display_df.columns:
                    display_df["Revenue"] = display_df["Revenue"].map(format_currency)
                if "Profit" in display_df.columns:
                    display_df["Profit"] = display_df["Profit"].map(format_currency)
                if "Anomaly Score" in display_df.columns:
                    display_df["Anomaly Score"] = pd.to_numeric(
                        display_df["Anomaly Score"], errors="coerce"
                    ).map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No other flagged incidents were sufficiently similar.")

            # =============================================================
            # VERIFIED TRANSACTION EVIDENCE
            # =============================================================
            st.divider()
            st.subheader("Verified Transaction Evidence")
            st.caption("These are original transaction fields returned by the analytics pipeline.")

            preferred_fields = [
                "order_id", "order_date", "customer_id", "region", "channel",
                "customer_segment", "category", "product", "quantity", "unit_price",
                "discount_pct", "revenue", "return_status", "return_amount",
                "delivery_days", "satisfaction_score", "shipping_cost",
                "operational_cost", "profit", "payment_method", "anomaly_score",
            ]

            evidence_rows = []
            for field in preferred_fields:
                if field not in row.index:
                    continue
                value = row[field]
                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass

                label = field.replace("_", " ").title()
                if field in {
                    "revenue", "return_amount", "shipping_cost",
                    "operational_cost", "profit", "unit_price",
                }:
                    display_value = format_currency(value)
                elif field == "discount_pct":
                    display_value = format_percent(value)
                else:
                    display_value = str(value)

                evidence_rows.append({"Field": label, "Verified Value": display_value})

            if evidence_rows:
                st.dataframe(
                    pd.DataFrame(evidence_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            # =============================================================
            # OPERATIONAL / FINANCIAL SIGNAL SNAPSHOT
            # =============================================================
            st.divider()
            st.subheader("Operational & Financial Snapshot")

            signal_fields = [
                ("Revenue", "revenue", "currency"),
                ("Profit", "profit", "currency"),
                ("Quantity", "quantity", "number"),
                ("Discount", "discount_pct", "percent"),
                ("Return Amount", "return_amount", "currency"),
                ("Delivery Days", "delivery_days", "number"),
                ("Satisfaction", "satisfaction_score", "number"),
                ("Shipping Cost", "shipping_cost", "currency"),
                ("Operational Cost", "operational_cost", "currency"),
            ]

            signal_cols = st.columns(3)
            for index, (label, field, kind) in enumerate(signal_fields):
                if field not in row.index:
                    continue
                value = _safe_float(row.get(field))
                if value is None:
                    continue
                with signal_cols[index % 3]:
                    if kind == "currency":
                        display = format_currency(value)
                    elif kind == "percent":
                        display = format_percent(value)
                    else:
                        display = f"{value:.2f}"
                    st.metric(label, display)

            # =============================================================
            # AI INVESTIGATOR
            # =============================================================
            st.divider()
            st.subheader("AI Investigator")
            st.caption(
                "Ask the local AI to investigate this specific incident. "
                "The prompt is constructed from verified transaction evidence, "
                "computed anomaly signals, and peer benchmarks."
            )

            investigator_questions = [
                "Explain why this incident deserves investigation.",
                "What are the strongest evidence-backed investigation hypotheses?",
                "Compare this incident with its peer transactions.",
                "What should an analyst verify next?",
            ]

            qcols = st.columns(2)
            for index, question_text in enumerate(investigator_questions):
                with qcols[index % 2]:
                    if st.button(
                        question_text,
                        key=f"incident_ai_suggestion_{index}_{order_id}",
                        use_container_width=True,
                    ):
                        st.session_state[f"incident_ai_question_{order_id}"] = question_text
                        st.rerun()

            custom_key = f"incident_ai_question_{order_id}"
            incident_question = st.text_area(
                "Investigation question",
                value=st.session_state.get(custom_key, ""),
                placeholder="Example: What should I verify before escalating this incident?",
                height=100,
                key=f"incident_ai_input_{order_id}",
            )

            if st.button(
                "Investigate with AI",
                type="primary",
                key=f"incident_ai_run_{order_id}",
                use_container_width=True,
            ):
                question = incident_question.strip()
                if not question:
                    st.warning("Enter an investigation question first.")
                else:
                    grounded_prompt = build_ai_investigation_prompt(
                        row,
                        drivers,
                        peer_comparisons,
                    )
                    with st.spinner("AI Investigator is reviewing the verified incident evidence..."):
                        result = ask_ai_assistant(
                            grounded_prompt + "\n\nANALYST QUESTION: " + question
                        )

                    if result.get("status") == "success":
                        answer = result.get("answer", "")
                        if answer:
                            st.markdown("### Investigation Finding")
                            st.write(answer)
                        else:
                            st.info("The AI Investigator returned an empty response.")
                    else:
                        st.error(
                            result.get(
                                "message",
                                "The AI Investigator could not answer the question.",
                            )
                        )

            # =============================================================
            # INVESTIGATION BOUNDARY + NEXT ACTION
            # =============================================================
            st.divider()
            st.subheader("Investigation Boundary")
            st.warning(
                "Isolation Forest identifies statistical unusualness. It does not establish "
                "fraud, malicious intent, policy violation, causation, or confirmed financial loss."
            )

            b1, b2 = st.columns(2)
            with b1:
                st.markdown("**Established by INSIGHT**")
                st.write("• This record is statistically unusual.")
                st.write("• The model score provides anomaly priority.")
                st.write("• Engineered signals identify unusual dimensions.")
                st.write("• Original transaction fields provide verifiable evidence.")
            with b2:
                st.markdown("**Requires human verification**")
                st.write("• Whether the transaction is actually fraudulent.")
                st.write("• Whether a customer or operational process caused the issue.")
                st.write("• Whether a policy was violated.")
                st.write("• Whether a real financial loss occurred.")

            st.subheader("Recommended Investigation Sequence")
            steps = [
                "Verify the underlying order, payment, and customer records.",
                "Check the incident against comparable transactions and historical behavior.",
                "Review fulfillment, return, delivery, and operational records where relevant.",
                "Document supporting evidence before escalating or taking action.",
            ]
            for index, step in enumerate(steps, start=1):
                st.markdown(f"**{index}.** {step}")

        else:
            # =============================================================
            # INCIDENT QUEUE
            # =============================================================
            st.subheader("Incident Queue")
            st.caption(
                "Incidents are ordered by anomaly score so the most statistically unusual "
                "records appear first. Select one to open the full investigation workspace."
            )

            display_limit = min(50, len(incident_df))
            for position, (index, row) in enumerate(
                incident_df.head(display_limit).iterrows(),
                start=1,
            ):
                order_id = str(row.get("order_id", f"INC-{position:04d}"))
                score = _safe_float(row.get("anomaly_score"))
                severity = "HIGH" if score is not None and score <= -0.15 else "MEDIUM"
                category = _safe_text(row.get("category"))
                region = _safe_text(row.get("region"))
                revenue = format_currency(row.get("revenue", 0))
                delivery = _safe_text(row.get("delivery_days"))
                return_status = _safe_text(row.get("return_status"))

                with st.container(border=True):
                    top_left, top_mid, top_right = st.columns([2.2, 2.2, 1])
                    with top_left:
                        st.markdown(f"### {severity} · {order_id}")
                        st.caption(f"{category} · {region}")
                    with top_mid:
                        st.write(f"**Anomaly Score:** {score:.4f}" if score is not None else "**Anomaly Score:** N/A")
                        st.write(f"**Revenue:** {revenue}")
                        st.caption(f"Delivery: {delivery} days · Return: {return_status}")
                    with top_right:
                        if st.button(
                            "Investigate →",
                            key=f"open_incident_{position}_{order_id}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_incident = index
                            st.rerun()

            if len(incident_df) > display_limit:
                st.caption(
                    f"Showing the {display_limit} highest-priority incidents by anomaly score."
                )

# =============================================================================
# RISK & ANOMALIES
# =============================================================================


elif selected_page == "Risk & Anomalies":

    st.header("Risk & Anomalies")

    st.write(
        "Potentially unusual transactions identified by "
        "the Isolation Forest anomaly detection model."
    )

    st.divider()

    evaluation = anomalies.get(
        "evaluation",
        {},
    ) if isinstance(anomalies, dict) else {}

    anomaly_records = anomalies.get(
        "predictions",
        [],
    ) if isinstance(anomalies, dict) else []

    anomaly_df_for_metrics = dataframe_from_value(
        anomaly_records
    )

    if (
        not anomaly_df_for_metrics.empty
        and "is_predicted_anomaly" in anomaly_df_for_metrics.columns
    ):
        anomaly_count = int(
            anomaly_df_for_metrics[
                "is_predicted_anomaly"
            ].sum()
        )
    else:
        anomaly_count = 0

    total_records = validation.get(
        "rows",
        10_000,
    )

    anomaly_rate = (
        anomaly_count / total_records * 100
        if total_records
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Detected Anomalies",
            format_number(
                anomaly_count
            ),
        )

    with col2:

        st.metric(
            "Anomaly Rate",
            format_percent(
                anomaly_rate
            ),
        )

    with col3:

        st.metric(
            "Analyzed Records",
            format_number(
                total_records
            ),
        )

    st.divider()

    # -------------------------------------------------------------------------
    # ANOMALY TABLE
    # -------------------------------------------------------------------------

    st.subheader(
        "Suspicious Transactions"
    )

    # The API returns anomaly predictions under "predictions".
    # Keep compatibility with older response keys as a fallback.
    if not anomaly_records:
        anomaly_records = anomalies.get(
            "records",
            anomalies.get(
                "top_anomalies",
                anomalies.get(
                    "anomalies",
                    [],
                ),
            ),
        )

    anomaly_df = dataframe_from_value(
        anomaly_records
    )

    if not anomaly_df.empty:

        st.dataframe(
            anomaly_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No anomaly records are currently available."
        )

    st.divider()

    # -------------------------------------------------------------------------
    # MODEL EVALUATION
    # -------------------------------------------------------------------------

    st.subheader(
        "Model Evaluation"
    )

    evaluation = anomalies.get(
        "evaluation",
        {},
    )

    if isinstance(
        evaluation,
        dict,
    ) and evaluation:

        eval_col1, eval_col2, eval_col3, eval_col4 = (
            st.columns(4)
        )

        with eval_col1:

            st.metric(
                "Precision",
                format_percent(
                    float(
                        evaluation.get(
                            "precision",
                            0,
                        )
                    )
                    * 100
                ),
            )

        with eval_col2:

            st.metric(
                "Recall",
                format_percent(
                    float(
                        evaluation.get(
                            "recall",
                            0,
                        )
                    )
                    * 100
                ),
            )

        with eval_col3:

            st.metric(
                "F1 Score",
                format_percent(
                    float(
                        evaluation.get(
                            "f1",
                            evaluation.get(
                                "f1_score",
                                0,
                            ),
                        )
                    )
                    * 100
                ),
            )

        with eval_col4:

            st.metric(
                "Actual Anomalies",
                format_number(
                    evaluation.get(
                        "actual_anomalies",
                        0,
                    )
                ),
            )

    else:

        st.caption(
            "Model evaluation metrics are not included in this API response."
        )

    st.info(
        "Ground-truth anomaly labels are used only for evaluation. "
        "They are not provided to the Isolation Forest model."
    )


# =============================================================================
# AI ASSISTANT
# =============================================================================


elif selected_page == "AI Assistant":

    st.header("AI Assistant")

    st.write(
        "Ask INSIGHT questions about the business data "
        "using natural language."
    )

    st.caption(
        "Analytics computes the evidence first. "
        "The local LLM explains the verified results."
    )

    st.divider()

    # -------------------------------------------------------------------------
    # EXAMPLE QUESTIONS
    # -------------------------------------------------------------------------

    st.subheader(
        "Suggested Questions"
    )

    examples = [
        "Which category has the highest return rate?",
        "Which region has the slowest delivery?",
        "Which channel has the lowest satisfaction?",
        "Which category is most profitable?",
        "Show me suspicious transactions.",
        "What should I investigate first?",
    ]

    example_columns = st.columns(3)

    for index, example in enumerate(
        examples
    ):

        with example_columns[
            index % 3
        ]:

            if st.button(
                example,
                key=f"example_question_{index}",
                use_container_width=True,
            ):

                st.session_state.ai_question = example

                st.rerun()

    st.divider()

    # -------------------------------------------------------------------------
    # CHAT HISTORY
    # -------------------------------------------------------------------------

    for message in st.session_state.ai_messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(
            role
        ):

            st.write(
                content
            )

    # -------------------------------------------------------------------------
    # QUESTION INPUT
    # -------------------------------------------------------------------------

    question = st.text_area(
        "Ask a business question",
        value=st.session_state.ai_question,
        placeholder=(
            "Example: Which category has the highest return rate?"
        ),
        height=110,
        key="ai_question_input",
    )

    ask_button = st.button(
        "Ask INSIGHT AI",
        type="primary",
        use_container_width=True,
    )

    if ask_button:

        current_question = (
            question.strip()
        )

        if not current_question:

            st.warning(
                "Please enter a question first."
            )

        else:

            # Store user message.

            st.session_state.ai_messages.append(
                {
                    "role": "user",
                    "content": current_question,
                }
            )

            with st.spinner(
                "Analyzing verified business evidence..."
            ):

                result = ask_ai_assistant(
                    current_question
                )

            if result.get(
                "status"
            ) == "success":

                answer = result.get(
                    "answer",
                    "",
                )

                route = result.get(
                    "route",
                    {},
                )

                route_category = route.get(
                    "category",
                    "general",
                )

                keywords = route.get(
                    "keywords",
                    [],
                )

                if not answer:

                    answer = (
                        "The AI Assistant returned "
                        "an empty response."
                    )

                st.session_state.ai_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                # Clear the example question after submission.

                st.session_state.ai_question = ""

                st.success(
                    "Analysis complete."
                )

                st.subheader(
                    "AI Analysis"
                )

                st.write(
                    answer
                )

                st.caption(
                    f"Question category: {route_category}"
                )

                if keywords:

                    st.caption(
                        "Detected keywords: "
                        + ", ".join(
                            str(keyword)
                            for keyword in keywords
                        )
                    )

            else:

                st.error(
                    result.get(
                        "message",
                        "The AI Assistant could not answer the question.",
                    )
                )


# =============================================================================
# FOOTER
# =============================================================================


st.divider()

st.caption(
    "INSIGHT · AI-Powered Decision Intelligence · "
    "Compute First, Explain Second"
)