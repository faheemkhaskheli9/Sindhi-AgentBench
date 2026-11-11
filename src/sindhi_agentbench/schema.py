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
    call_id: str | None = None


class Task(StrictModel):
    id: str = Field(min_length=1)
    language: Language
    domain: Domain
    prompt: str = Field(min_length=1)
    expected_call: ToolCall | None
    expected_behavior: Literal["execute", "clarify", "refuse"] = "execute"
    expected_facts: dict[str, Any] = Field(default_factory=dict)
    verification_status: Literal["unverified", "verified"] = "unverified"
    annotator_id: str | None = None
    reviewer_id: str | None = None

    @model_validator(mode="after")
    def require_expected_action_consistency(self) -> "Task":
        if self.expected_behavior == "execute" and self.expected_call is None:
            raise ValueError("Executable tasks require expected_call")
        if self.expected_behavior != "execute" and (
            self.expected_call is not None or self.expected_facts
        ):
            raise ValueError("Clarification/refusal tasks cannot require a tool or result facts")
        return self

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
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str = ""

    @model_validator(mode="after")
    def calls_must_agree(self) -> "Prediction":
        if self.tool_call is not None and self.tool_calls and (
            len(self.tool_calls) != 1 or self.tool_call != self.tool_calls[0]
        ):
            raise ValueError("tool_call and tool_calls conflict")
        return self


class HumanReview(StrictModel):
    task_id: str
    reviewer_id: str = Field(min_length=1)
    language_fidelity: float = Field(ge=0, le=1)
    hallucinated: bool
    behavior_correct: bool | None = None


class TaskScore(StrictModel):
    task_id: str
    language: Language
    tool_selection: float
    parameter_accuracy: float | None
    task_completion: float | None
    behavior_correct: float | None = None
    language_fidelity: float | None = None
    hallucination_rate: float | None = None
    simulated_result: dict[str, Any] | None = None


class Report(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    evaluated_tasks: int
    by_language: dict[str, dict[str, float | int | None]]
    overall: dict[str, float | int | None]
    scores: list[TaskScore]
