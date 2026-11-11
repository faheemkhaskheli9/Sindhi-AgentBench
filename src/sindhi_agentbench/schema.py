"""Versioned, explicit JSONL contracts for tasks, predictions, and reviews."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Language(str, Enum):
    SINDHI = "sd"
    ROMAN_SINDHI = "sd-Latn"
    URDU = "ur"
    ENGLISH = "en"


class Domain(str, Enum):
    BANKING = "banking"
    BOOKING = "booking"
    SHOPPING = "shopping"
    GOVERNMENT = "government"


class ToolCall(StrictModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class Task(StrictModel):
    id: str = Field(min_length=1)
    language: Language
    domain: Domain
    prompt: str = Field(min_length=1)
    expected_call: ToolCall
    expected_facts: dict[str, Any] = Field(default_factory=dict)
    verification_status: Literal["unverified", "verified"] = "unverified"
    annotator_id: str | None = None
    reviewer_id: str | None = None

    @model_validator(mode="after")
    def verified_requires_independent_review(self) -> "Task":
        if self.verification_status == "verified":
            if not self.annotator_id or not self.reviewer_id:
                raise ValueError("Verified tasks need annotator_id and reviewer_id")
            if self.annotator_id == self.reviewer_id:
                raise ValueError("Verifier must differ from annotator")
        return self


class Prediction(StrictModel):
    task_id: str
    tool_call: ToolCall | None = None
    final_answer: str = ""


class HumanReview(StrictModel):
    task_id: str
    reviewer_id: str = Field(min_length=1)
    language_fidelity: float = Field(ge=0, le=1)
    hallucinated: bool


class TaskScore(StrictModel):
    task_id: str
    language: Language
    tool_selection: float
    parameter_accuracy: float
    task_completion: float
    language_fidelity: float | None = None
    hallucination_rate: float | None = None
    simulated_result: dict[str, Any] | None = None


class Report(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    evaluated_tasks: int
    by_language: dict[str, dict[str, float | int | None]]
    overall: dict[str, float | int | None]
    scores: list[TaskScore]
