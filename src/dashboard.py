import requests
import streamlit as st


API_URL = "http://insight:8080"


st.set_page_config(
    page_title="INSIGHT",
    page_icon="I",
    layout="wide",
)


def get_analysis():
    """
    Fetch the verified analysis from the INSIGHT API.
    """

    response = requests.get(
        f"{API_URL}/api/analyze-synthetic",
        timeout=300,
    )

    response.raise_for_status()

    return response.json()


def format_currency(value):
    """
    Format a numeric value as Indian currency.
    """

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.2f}K"

    return f"₹{value:.2f}"


def display_insight(insight):
    """
    Display one evidence-backed business insight.
    """

    severity = insight["severity"]

    if severity == "high":
        st.error(
            f"🔴 HIGH — {insight['title']}"
        )
    elif severity == "medium":
        st.warning(
            f"🟡 MEDIUM — {insight['title']}"
        )
    else:
        st.info(
            f"🔵 LOW — {insight['title']}"
        )

    st.markdown(
        f"**Finding:** {insight['finding']}"
    )

    st.markdown(
        f"**Impact:** {insight['impact']}"
    )

    with st.expander("View verified evidence"):
        for evidence in insight["evidence"]:
            st.json(evidence)

    st.markdown("**Possible contributing factors**")

    for factor in insight["possible_contributing_factors"]:
        st.write(f"• {factor}")

    st.markdown("**Recommendation**")

    st.success(
        insight["recommendation"]
    )


def main():
    st.title("INSIGHT")

    st.subheader(
        "AI-Powered Decision Intelligence Platform"
    )

    st.write(
        "Transform business data into verified findings, "
        "explanations, and decisions."
    )

    st.divider()

    with st.spinner(
        "Loading business intelligence..."
    ):
        try:
            result = get_analysis()

        except requests.RequestException as exc:
            st.error(
                "Unable to connect to the INSIGHT API."
            )
            st.code(str(exc))
            return

        except Exception as exc:
            st.error(
                "Unable to load the analysis."
            )
            st.code(str(exc))
            return

    if result.get("status") != "success":
        st.error(
            "The analysis service did not return a successful result."
        )
        return

    kpis = result["kpis"]

    # ---------------------------------------------------------
    # Dataset status
    # ---------------------------------------------------------

    st.caption(
        f"Dataset: {result['filename']} · "
        f"{result['rows']:,} records · "
        f"{result['columns']} columns"
    )

    # ---------------------------------------------------------
    # KPI overview
    # ---------------------------------------------------------

    st.header("Business Overview")

    row_1 = st.columns(4)

    with row_1[0]:
        st.metric(
            "Revenue",
            format_currency(
                kpis["total_revenue"]
            ),
        )

    with row_1[1]:
        st.metric(
            "Orders",
            f"{kpis['total_orders']:,}",
        )

    with row_1[2]:
        st.metric(
            "Customers",
            f"{kpis['total_customers']:,}",
        )

    with row_1[3]:
        st.metric(
            "Profit",
            format_currency(
                kpis["total_profit"]
            ),
        )

    row_2 = st.columns(4)

    with row_2[0]:
        st.metric(
            "Profit Margin",
            f"{kpis['profit_margin']:.2f}%",
        )

    with row_2[1]:
        st.metric(
            "Return Rate",
            f"{kpis['return_rate']:.2f}%",
        )

    with row_2[2]:
        st.metric(
            "Avg Delivery",
            f"{kpis['average_delivery_time']:.2f} days",
        )

    with row_2[3]:
        st.metric(
            "Satisfaction",
            f"{kpis['average_satisfaction']:.2f}/5",
        )

    st.divider()

    # ---------------------------------------------------------
    # Insights
    # ---------------------------------------------------------

    st.header("Verified Business Insights")

    st.caption(
        "Insights are generated from computed analytics "
        "and machine-learning outputs."
    )

    for insight in result["insights"]:
        display_insight(insight)
        st.divider()

    # ---------------------------------------------------------
    # System status
    # ---------------------------------------------------------

    st.header("Analysis Status")

    status_col_1, status_col_2 = st.columns(2)

    with status_col_1:
        st.success(
            "✓ Analytics pipeline completed"
        )

    with status_col_2:
        st.success(
            f"✓ {result['insight_count']} insights generated"
        )


if __name__ == "__main__":
    main()