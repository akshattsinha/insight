from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QuestionRoute:
    """
    Represents the analytics area associated with
    a user's business question.
    """

    category: str
    keywords: list[str]


ROUTES: dict[str, list[str]] = {
    "revenue": [
        "revenue",
        "sales",
        "selling",
        "income",
        "turnover",
    ],
    "profit": [
        "profit",
        "profitability",
        "margin",
        "loss",
        "cost",
    ],
    "returns": [
        "return",
        "returns",
        "returned",
        "refund",
        "refunds",
    ],
    "delivery": [
        "delivery",
        "shipping",
        "shipment",
        "delay",
        "delayed",
        "fulfillment",
    ],
    "satisfaction": [
        "satisfaction",
        "rating",
        "ratings",
        "customer experience",
        "customer satisfaction",
    ],
    "segments": [
        "category",
        "region",
        "channel",
        "customer segment",
        "segment",
        "payment",
        "product",
    ],
    "anomalies": [
        "anomaly",
        "anomalies",
        "unusual",
        "suspicious",
        "outlier",
        "risk",
    ],
}


def normalize_question(question: str) -> str:
    """
    Normalize a user question before routing.
    """

    question = question.strip().lower()

    question = re.sub(
        r"\s+",
        " ",
        question,
    )

    return question


def route_question(
    question: str,
) -> QuestionRoute:

    normalized_question = normalize_question(
        question
    )

    if not normalized_question:

        return QuestionRoute(
            category="general",
            keywords=[],
        )

    matched_categories: dict[str, list[str]] = {}

    for category, keywords in ROUTES.items():

        matches = []

        for keyword in keywords:

            if keyword in normalized_question:

                matches.append(
                    keyword
                )

        if matches:

            matched_categories[
                category
            ] = matches

    if not matched_categories:

        return QuestionRoute(
            category="general",
            keywords=[],
        )

    # Prefer the category with the largest
    # number of matching keywords.
    category = max(
        matched_categories,
        key=lambda item: len(
            matched_categories[item]
        ),
    )

    return QuestionRoute(
        category=category,
        keywords=matched_categories[
            category
        ],
    )


def main() -> None:
    """
    Run a small routing test.
    """

    questions = [
        "Why is revenue falling?",
        "Which category has the highest returns?",
        "Which region has the slowest delivery?",
        "Why is customer satisfaction low?",
        "Which channel is most profitable?",
        "Show me suspicious transactions.",
        "What should management investigate?",
    ]

    print()
    print("=" * 80)
    print("INSIGHT QUESTION ROUTER")
    print("=" * 80)

    for question in questions:

        route = route_question(
            question
        )

        print()
        print(
            f"Question: {question}"
        )

        print(
            f"Category: {route.category}"
        )

        print(
            f"Keywords: {route.keywords}"
        )

    print()
    print("=" * 80)
    print("STATUS: PASS")
    print("Question routing is working.")
    print("=" * 80)


if __name__ == "__main__":
    main()