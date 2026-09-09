from __future__ import annotations

from typing import Any

from src.chatbot.context import AnalyticsContextBuilder
from src.chatbot.router import QuestionRoute, route_question
from src.llm.explainer import InsightExplainer


class InsightChatbot:
    """
    Coordinates question routing, analytical context construction,
    and LLM explanation.

    The chatbot uses the verified analysis supplied by the main
    analytics pipeline whenever it is available.
    """

    def __init__(
        self,
        context_builder: AnalyticsContextBuilder | None = None,
        explainer: InsightExplainer | None = None,
    ):
        self.context_builder = (
            context_builder or AnalyticsContextBuilder()
        )

        self.explainer = (
            explainer or InsightExplainer()
        )

    def ask(
        self,
        question: str,
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Answer a business question using verified analytical evidence.

        When analysis is supplied, the chatbot reuses the already
        computed INSIGHT analytics instead of independently loading
        and analyzing the synthetic dataset.
        """

        route: QuestionRoute = route_question(question)

        context = self.context_builder.build_context(
            route=route,
            analysis=analysis,
        )

        insights = []

        if analysis is not None:
            insights = analysis.get(
                "insights",
                [],
            )

        selected_insight = self._select_insight(
            question=question,
            route=route,
            insights=insights,
        )

        if selected_insight is None:
            return {
                "response": (
                    "I could not find a verified insight in the "
                    "current analysis that directly answers this "
                    "question."
                ),
                "category": route.category,
                "keywords": route.keywords,
                "context": context,
            }

        try:
            explanation = self.explainer.explain(
                selected_insight,
            )

        except Exception as exc:
            return {
                "response": (
                    "The verified analysis was found, but the "
                    "AI explanation could not be generated: "
                    f"{exc}"
                ),
                "category": route.category,
                "keywords": route.keywords,
                "context": context,
                "insight": self._serialize_insight(
                    selected_insight,
                ),
            }

        return {
            "response": explanation,
            "category": route.category,
            "keywords": route.keywords,
            "context": context,
            "insight": self._serialize_insight(
                selected_insight,
            ),
        }

    @staticmethod
    def _select_insight(
        question: str,
        route: QuestionRoute,
        insights: list[Any],
    ) -> Any | None:
        """
        Select the most relevant verified Insight.
        """

        if not insights:
            return None

        normalized_question = question.lower()

        category_terms = {
            route.category.lower(),
            *[
                keyword.lower()
                for keyword in route.keywords
            ],
        }

        best_insight = None
        best_score = -1

        for insight in insights:

            title = str(
                getattr(
                    insight,
                    "title",
                    "",
                )
            ).lower()

            finding = str(
                getattr(
                    insight,
                    "finding",
                    "",
                )
            ).lower()

            recommendation = str(
                getattr(
                    insight,
                    "recommendation",
                    "",
                )
            ).lower()

            searchable_text = (
                f"{title} "
                f"{finding} "
                f"{recommendation}"
            )

            score = 0

            if route.category.lower() in searchable_text:
                score += 3

            for keyword in category_terms:
                if keyword and keyword in searchable_text:
                    score += 2

            for word in normalized_question.split():
                if (
                    len(word) >= 4
                    and word in searchable_text
                ):
                    score += 1

            if score > best_score:
                best_score = score
                best_insight = insight

        return best_insight

    @staticmethod
    def _serialize_insight(
        insight: Any,
    ) -> dict[str, Any]:
        """
        Convert an Insight object into a JSON-safe dictionary.
        """

        if hasattr(insight, "model_dump"):
            return insight.model_dump()

        if hasattr(insight, "dict"):
            return insight.dict()

        if isinstance(insight, dict):
            return insight

        return {
            "title": getattr(
                insight,
                "title",
                "",
            ),
            "finding": getattr(
                insight,
                "finding",
                "",
            ),
            "evidence": getattr(
                insight,
                "evidence",
                [],
            ),
            "severity": getattr(
                insight,
                "severity",
                "",
            ),
            "impact": getattr(
                insight,
                "impact",
                "",
            ),
            "possible_contributing_factors": getattr(
                insight,
                "possible_contributing_factors",
                [],
            ),
            "recommendation": getattr(
                insight,
                "recommendation",
                "",
            ),
        }