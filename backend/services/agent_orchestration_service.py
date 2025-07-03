from __future__ import annotations

"""Service that orchestrates multiple LLM "agents" to process a natural-language query.

Pipeline
---------
1. Pre-Processor Agent – rewrites / clarifies the user phrasing.
2. Translator Agent – delegates to the existing ``TranslationService`` (NL → GraphQL).
3. Reviewer Agent   – inspects the GraphQL + original query and provides feedback / validation.

The class is intentionally **stateless** so it can be instantiated per request or injected
into FastAPI routes without side-effects.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from services.ollama_service import OllamaService, GenerationResult
from services.translation_service import TranslationService, TranslationResult

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Structured output from the reviewer agent."""

    passed: bool
    comments: List[str]
    suggested_improvements: List[str]
    raw_response: str


@dataclass
class MultiAgentResult:
    """Full artefacts returned by the orchestration pipeline."""

    original_query: str
    rewritten_query: str
    translation: TranslationResult
    review: ReviewResult
    processing_time: float

    def to_json(self) -> str:
        """Handy helper for pretty-printing / serialisation."""
        return json.dumps(asdict(self), indent=2, default=str)


class AgentOrchestrationService:
    """Coordinates several LLMs (agents) to improve translation accuracy."""

    def __init__(self):
        self.settings = get_settings()
        self.ollama_service = OllamaService()
        self.translation_service = TranslationService()

    # ---------------------------------------------------------------------
    # INTERNAL PROMPT BUILDERS
    # ---------------------------------------------------------------------
    def _build_pre_prompt(self) -> str:
        """System prompt for the pre-processing agent."""
        return (
            "You are an expert technical writer. Rephrase the user query below so that "
            "an LLM can easily translate it to an accurate GraphQL query. "
            "Clarify pronouns, fill in implied context, and expand abbreviations. "
            "Return ONLY the rewritten query as plain text. Do NOT wrap in JSON or markdown."
        )

    def _build_review_prompt(self, original_query: str, graphql_query: str) -> str:
        """System prompt for the reviewer agent."""
        return (
            "You are a senior GraphQL reviewer. Compare the original natural language "
            "request and the generated GraphQL query. Assess if the query satisfies the "
            "intent, spot security or performance issues, and propose improvements.\n\n"
            f"Original query:\n{original_query}\n\n"
            f"GraphQL proposal:\n{graphql_query}\n\n"
            "Respond in JSON with this schema:\n"
            "{\n  \"passed\": boolean,\n  \"comments\": string[],\n  \"suggested_improvements\": string[]\n}"
        )

    # ---------------------------------------------------------------------
    # AGENT HELPERS
    # ---------------------------------------------------------------------
    async def _call_llm(self, messages: List[Dict[str, str]], model: str) -> GenerationResult:
        """Tiny wrapper around ``OllamaService.chat_completion`` with logging."""
        start = time.time()
        result = await self.ollama_service.chat_completion(messages=messages, model=model)
        logger.debug(
            f"LLM call (model={model}) took {time.time() - start:.2f}s and returned "
            f"{len(result.text)} chars"
        )
        return result

    async def _rewrite_query(self, query: str, model: str) -> str:
        system_prompt = self._build_pre_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        response = await self._call_llm(messages, model)
        return response.text.strip().strip("`")  # remove accidental code fencing

    async def _review_translation(
        self, *, original_query: str, graphql_query: str, model: str
    ) -> ReviewResult:
        system_prompt = self._build_review_prompt(original_query, graphql_query)
        messages = [{"role": "system", "content": system_prompt}]
        response = await self._call_llm(messages, model)

        # Attempt to parse JSON from the response. Fall back gracefully.
        try:
            parsed = json.loads(response.text.strip().strip("`"))
            passed = bool(parsed.get("passed"))
            comments = parsed.get("comments", [])
            suggestions = parsed.get("suggested_improvements", [])
        except json.JSONDecodeError:
            logger.warning("Reviewer returned non-JSON response; treating as comments only.")
            passed = False
            comments = [response.text.strip()]
            suggestions = []

        return ReviewResult(
            passed=passed,
            comments=comments,
            suggested_improvements=suggestions,
            raw_response=response.text,
        )

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------
    async def process_query(
        self,
        query: str,
        *,
        pre_model: Optional[str] = None,
        translator_model: Optional[str] = None,
        review_model: Optional[str] = None,
    ) -> MultiAgentResult:
        """Run the full pipeline and return structured results."""
        overall_start = time.time()

        pre_model = pre_model or self.settings.ollama.default_model
        translator_model = translator_model or self.settings.ollama.default_model
        review_model = review_model or self.settings.ollama.default_model

        # 1️⃣ Pre-process / rewrite
        rewritten_query = await self._rewrite_query(query, pre_model)
        logger.info(f"[Pre-Processor] → {rewritten_query}")

        # 2️⃣ Translate to GraphQL
        translation_result = await self.translation_service.translate_to_graphql(
            rewritten_query, model=translator_model
        )
        logger.info("[Translator] GraphQL generated with confidence %.2f", translation_result.confidence)

        # 3️⃣ Review / validate
        review_result = await self._review_translation(
            original_query=query,
            graphql_query=translation_result.graphql_query or "",
            model=review_model,
        )
        logger.info("[Reviewer] passed=%s (%d comments)", review_result.passed, len(review_result.comments))

        processing_time = time.time() - overall_start

        return MultiAgentResult(
            original_query=query,
            rewritten_query=rewritten_query,
            translation=translation_result,
            review=review_result,
            processing_time=processing_time,
        ) 