"""RubrifyColumnConfig -- DataDesigner column config for rubrify-based judging.

This config tells DataDesigner how to run a rubrify rubric evaluation
against a target column (and optional context column), producing
per-row judgments and an overall pass/fail decision as side-effect
columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from data_designer.config.base import SingleColumnConfig


class RubrifyColumnConfig(SingleColumnConfig):
    """Column config that runs a rubrify rubric evaluation.

    Evaluates the contents of ``target_column`` against a compiled
    rubrify ``RubricBundle``.  The bundle can be supplied either as a
    path to a JSON file (``rubric_path``) or as inline JSON
    (``rubric_json``).  Exactly one of the two must be provided.

    Side-effect columns created:
        ``{name}__judgments`` -- per-criterion judgment details (JSON).
        ``{name}__decision`` -- overall pass/fail decision string.

    Attributes:
        target_column: Name of the column whose cell values are evaluated.
        model_alias: Alias of the model configuration to use as the judge
            LLM.  Must match an alias defined when initializing the
            DataDesignerConfigBuilder.
        rubric_path: File-system path to a compiled rubric bundle JSON.
            Relative paths are resolved against the current working directory.
            Mutually exclusive with ``rubric_json``.
        rubric_json: Inline compiled rubric bundle as a JSON string.
            Mutually exclusive with ``rubric_path``.
        context_column: Optional column supplying additional context for
            the judge (e.g. the original prompt or instruction).
        genre: Optional genre tag used to filter applicable criteria
            within the rubric.
        judge_temperature: Sampling temperature for the judge model.
        judge_max_tokens: Maximum tokens the judge model may generate.
        parallel_criteria: If True, evaluate criteria concurrently rather
            than sequentially.
    """

    column_type: Literal["rubrify-judge"] = "rubrify-judge"

    # ── required ──────────────────────────────────────────────────
    target_column: str = Field(
        description="Name of the column to evaluate with the rubric.",
    )
    model_alias: str = Field(
        description="Alias of the model configuration to use as the judge LLM.",
    )

    # ── rubric source (exactly one required) ──────────────────────
    rubric_path: str | None = Field(
        default=None,
        description="Path to a compiled rubric bundle JSON file.",
    )
    rubric_json: str | None = Field(
        default=None,
        description="Inline compiled rubric bundle as a JSON string.",
    )

    # ── optional ──────────────────────────────────────────────────
    context_column: str | None = Field(
        default=None,
        description="Optional column supplying additional context for the judge.",
    )
    genre: str | None = Field(
        default=None,
        description="Optional genre tag to filter applicable criteria.",
    )
    judge_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for the judge model.",
    )
    judge_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens the judge model may generate.",
    )
    parallel_criteria: bool = Field(
        default=False,
        description="If True, evaluate criteria concurrently rather than sequentially.",
    )

    # ── validators ────────────────────────────────────────────────

    @model_validator(mode="after")
    def _exactly_one_rubric_source(self) -> Self:
        """Ensure exactly one of rubric_path / rubric_json is provided."""
        has_path = self.rubric_path is not None
        has_json = self.rubric_json is not None
        if has_path == has_json:
            raise ValueError(
                "Exactly one of 'rubric_path' or 'rubric_json' must be provided, "
                f"got {'both' if has_path else 'neither'}."
            )
        return self

    @field_validator("rubric_path", mode="after")
    @classmethod
    def _validate_rubric_path(cls, v: str | None) -> str | None:
        if v is None:
            return v
        path = Path(v)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise ValueError(f"rubric_path does not exist: {path}")
        if path.suffix != ".json":
            raise ValueError(
                f"rubric_path must point to a .json file, got: {path.suffix!r}"
            )
        return str(path)

    @field_validator("rubric_json", mode="after")
    @classmethod
    def _validate_rubric_json(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            json.loads(v)
        except json.JSONDecodeError as exc:
            raise ValueError(f"rubric_json is not valid JSON: {exc}") from exc
        return v

    # ── abstract property implementations ─────────────────────────

    @property
    def required_columns(self) -> list[str]:
        """Columns that must exist before this column can be generated."""
        cols = [self.target_column]
        if self.context_column is not None:
            cols.append(self.context_column)
        return cols

    @property
    def side_effect_columns(self) -> list[str]:
        """Additional columns created alongside the primary column."""
        return [
            f"{self.name}__judgments",
            f"{self.name}__decision",
        ]

    # ── optional overrides ────────────────────────────────────────

    @staticmethod
    def get_column_emoji() -> str:
        return "\u2696\ufe0f"  # balance scale
