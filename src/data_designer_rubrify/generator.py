"""RubrifyCellGenerator -- DataDesigner column generator for rubrify judging.

Bridges DataDesigner's column generation pipeline to rubrify's Judge.
Each row is evaluated independently against a compiled RubricBundle,
producing a normalized score, per-criterion judgments, and a decision.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorWithModelRegistry,
)

from harn_ai.models import get_model as harn_get_model
from harn_ai.env_api_keys import get_env_api_key
from harn_ai.types import Model, ModelCost

from rubrify.ir.bundle import RubricBundle
from rubrify.engine.judge import Judge, JudgeConfig
from rubrify.engine.judgment import Judgment

from data_designer_rubrify.config import RubrifyColumnConfig

logger = logging.getLogger(__name__)


def _rehydrate_compiled_patterns(bundle: RubricBundle) -> RubricBundle:
    """Rehydrate compiled_patterns on a deserialized RubricBundle.

    ``compiled_patterns`` is excluded from JSON serialization (``exclude=True``
    on the field), so after round-tripping through JSON it will be empty.
    We rebuild it from the rubric's ``patterns`` (PatternEntry) and
    ``disqualifiers`` (Disqualifier.pattern), using the same logic as
    ``lock_bundle()``.

    Because RubricBundle is frozen, we must construct a new instance with
    the rehydrated patterns dict.
    """
    compiled: dict[str, re.Pattern[str]] = {}

    for p in bundle.rubric.patterns:
        flags = re.IGNORECASE if "i" in p.flags else 0
        try:
            compiled[p.id] = re.compile(p.pattern, flags)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex in PatternEntry '{p.id}': {exc}"
            ) from exc

    for dq in bundle.rubric.disqualifiers:
        if dq.pattern:
            try:
                compiled[f"dq_{dq.id}"] = re.compile(dq.pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex in Disqualifier '{dq.id}': {exc}"
                ) from exc

    if not compiled:
        # No patterns to rehydrate -- return the bundle as-is.
        return bundle

    # RubricBundle is frozen, so we reconstruct with the populated dict.
    return RubricBundle(
        rubric=bundle.rubric,
        compiled_patterns=compiled,
        bindings=bundle.bindings,
        authority_blocks=bundle.authority_blocks,
        surface_policy=bundle.surface_policy,
        output_constraints=bundle.output_constraints,
        locked=bundle.locked,
    )


def _build_harn_model(provider_name: str, model_id: str, base_url: str) -> Model:
    """Build a harn_ai Model for the judge.

    First tries the built-in catalog via ``harn_ai.models.get_model``.
    Falls back to constructing a minimal Model manually for custom/private
    endpoints that are not in the catalog.
    """
    catalog_model = harn_get_model(provider_name, model_id)
    if catalog_model is not None:
        return catalog_model

    # Fallback: construct a minimal Model for custom endpoints.
    # Map common provider names to their API format.
    _PROVIDER_API_MAP: dict[str, str] = {
        "openai": "openai-completions",
        "anthropic": "anthropic-messages",
        "google": "google-generative-ai",
        "google-vertex": "google-vertex",
        "mistral": "mistral-conversations",
        "deepseek": "openai-completions",
        "groq": "openai-completions",
        "cerebras": "openai-completions",
        "xai": "openai-completions",
        "openrouter": "openai-completions",
        "fireworks": "openai-completions",
        "together": "openai-completions",
    }
    api = _PROVIDER_API_MAP.get(provider_name, "openai-completions")

    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider=provider_name,
        baseUrl=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cacheRead=0.0, cacheWrite=0.0),
        contextWindow=128_000,
        maxTokens=4096,
    )


def _resolve_api_key(
    dd_provider_api_key: str | None,
    provider_name: str,
) -> str | None:
    """Resolve an API key for the judge model.

    Priority:
      1. Explicit key from DD's ModelProvider config.
      2. Environment-based discovery via harn's ``get_env_api_key``.
    """
    if dd_provider_api_key:
        return dd_provider_api_key
    return get_env_api_key(provider_name)


class RubrifyCellGenerator(ColumnGeneratorWithModelRegistry[RubrifyColumnConfig]):
    """Cell-by-cell generator that evaluates each row with a rubrify Judge.

    Produces three outputs per row:
      - ``{name}``             -- normalized aggregate score (float)
      - ``{name}__judgments``   -- per-criterion judgment details (JSON string)
      - ``{name}__decision``   -- overall pass/fail decision string
    """

    # ── ConfigurableTask lifecycle ────────────────────────────────────

    def _initialize(self) -> None:
        """Called by ConfigurableTask.__init__() after config validation.

        Loads the rubric bundle, rehydrates compiled patterns, builds
        the harn_ai Model from DD's model config, and constructs the
        Judge instance.
        """
        # 1. Load the RubricBundle
        self._bundle = self._load_bundle()

        # 2. Rehydrate compiled_patterns (excluded from JSON serialization)
        self._bundle = _rehydrate_compiled_patterns(self._bundle)

        # 3. Build the harn_ai Model from DD's model config
        model_config = self.get_model_config(model_alias=self.config.model_alias)
        provider_name = self.get_model_provider_name(
            model_alias=self.config.model_alias,
        )

        # Get the provider's base URL for custom endpoint fallback
        dd_provider = self.model_registry.get_model_provider(
            model_alias=self.config.model_alias,
        )
        base_url = dd_provider.endpoint

        harn_model = _build_harn_model(
            provider_name=provider_name,
            model_id=str(model_config.model),
            base_url=base_url,
        )

        # 4. Resolve API key
        api_key = _resolve_api_key(
            dd_provider_api_key=dd_provider.api_key,
            provider_name=provider_name,
        )

        # 5. Construct the Judge
        self._judge = Judge(
            JudgeConfig(
                model=harn_model,
                api_key=api_key,
                temperature=self.config.judge_temperature,
                max_tokens=self.config.judge_max_tokens,
                parallel=self.config.parallel_criteria,
            )
        )

        logger.info(
            "Initialized RubrifyCellGenerator for column %r "
            "(model=%r, provider=%r, rubric=%r)",
            self.config.name,
            str(model_config.model),
            provider_name,
            self._bundle.rubric.meta.name,
        )

    # ── bundle loading ────────────────────────────────────────────────

    def _load_bundle(self) -> RubricBundle:
        """Load and validate the RubricBundle from config source."""
        if self.config.rubric_path is not None:
            raw = Path(self.config.rubric_path).read_text(encoding="utf-8")
        elif self.config.rubric_json is not None:
            raw = self.config.rubric_json
        else:
            # Should never happen -- config validator enforces exactly one
            raise ValueError(
                "Neither rubric_path nor rubric_json provided. "
                "This should have been caught by config validation."
            )

        data = json.loads(raw)
        bundle = RubricBundle.model_validate(data)

        if not bundle.locked:
            raise ValueError(
                "RubricBundle is not locked. Only locked (compiled) bundles "
                "can be used for evaluation. Run the rubrify compiler first."
            )

        return bundle

    # ── generation strategy ───────────────────────────────────────────

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    # ── async generation (the engine calls this per-row) ──────────────

    async def agenerate(self, data: dict) -> dict:
        """Evaluate a single row against the rubric bundle.

        Extracts the target text (and optional context) from the row,
        runs the judge, and packs the results into the output dict.
        """
        score_col = self.config.name
        judgments_col = f"{self.config.name}__judgments"
        decision_col = f"{self.config.name}__decision"

        try:
            # Extract inputs
            response_text = data.get(self.config.target_column)

            # Handle empty/null input gracefully
            if response_text is None or (
                isinstance(response_text, str) and not response_text.strip()
            ):
                data[score_col] = None
                data[judgments_col] = None
                data[decision_col] = None
                return data

            response_text = str(response_text)

            # Extract optional context
            context_text: str | None = None
            if self.config.context_column is not None:
                raw_context = data.get(self.config.context_column)
                if raw_context is not None:
                    context_text = str(raw_context)

            # Run the judge
            judgment: Judgment = await self._judge.evaluate(
                self._bundle,
                response_text,
                context_text=context_text,
                genre=self.config.genre,
            )

            # Pack results
            data[score_col] = judgment.aggregation.normalized_score

            # Serialize per-criterion judgments to JSON string
            criterion_dicts = [
                cj.model_dump(mode="json")
                for cj in judgment.criterion_judgments
            ]
            data[judgments_col] = json.dumps(criterion_dicts)

            data[decision_col] = judgment.decision

        except Exception:
            logger.exception(
                "Error evaluating row for column %r; returning None values",
                self.config.name,
            )
            data[score_col] = None
            data[judgments_col] = None
            data[decision_col] = None

        return data

    # ── logging ───────────────────────────────────────────────────────

    def log_pre_generation(self) -> None:
        model_config = self.get_model_config(model_alias=self.config.model_alias)
        provider_name = self.get_model_provider_name(
            model_alias=self.config.model_alias,
        )
        logger.info(
            "%s rubrify-judge config for column '%s'",
            self.config.get_column_emoji(),
            self.config.name,
        )
        logger.info("  model: %r", str(model_config.model))
        logger.info("  model alias: %r", self.config.model_alias)
        logger.info("  model provider: %r", provider_name)
        logger.info("  rubric: %r", self._bundle.rubric.meta.name)
        logger.info(
            "  criteria: %d", len(self._bundle.rubric.criteria)
        )
        logger.info("  target column: %r", self.config.target_column)
        if self.config.context_column:
            logger.info("  context column: %r", self.config.context_column)
        if self.config.genre:
            logger.info("  genre: %r", self.config.genre)
