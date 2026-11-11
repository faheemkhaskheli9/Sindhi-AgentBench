"""Provider-neutral, bounded multi-turn agent execution over offline mock APIs.

The adapter never receives expected actions or outcomes. No live transactions occur.
"""

import json
from enum import Enum
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from sindhi_agentbench.environments import TOOL_SCHEMAS, ToolResult, execute_mock
from sindhi_agentbench.schema import Prediction, Task, ToolCall


class BaseContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseContract):
    role: Role
    content: str
    tool_name: str | None = None
    tool_call: ToolCall | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ToolSpec(BaseContract):
    name: str
    description: str
    parameters: dict


class ModelTurn(BaseContract):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class AgentAdapter(Protocol):
    """Implement this for each model provider; API credentials stay adapter-side."""

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn: ...


class ExecutedAction(BaseContract):
    call: ToolCall
    result: ToolResult


class Termination(str, Enum):
    COMPLETE = "complete"
    MAX_TURNS = "max_turns"
    ADAPTER_ERROR = "adapter_error"


class AgentRun(BaseContract):
    task_id: str
    messages: list[Message]
    actions: list[ExecutedAction]
    final_answer: str
    termination: Termination
    elapsed_ms: float = Field(ge=0)
    error_type: str | None = None

    def as_prediction(self) -> Prediction:
        # Single-action tasks must not reward extra or repeated tool calls.
        call = self.actions[0].call if len(self.actions) == 1 else None
        answer = self.final_answer if self.termination == Termination.COMPLETE else ""
        return Prediction(task_id=self.task_id, tool_call=call, final_answer=answer)


def tool_specs() -> list[ToolSpec]:
    descriptions = {
        "bank.transfer": "Transfer fictional PKR between sandbox accounts.",
        "booking.reserve": "Reserve a fictional transport service in a sandbox.",
        "shopping.place_order": "Place a fictional order in a sandbox.",
        "government.check_application": "Check a fictional government application status.",
    }
    return [
        ToolSpec(name=name, description=descriptions[name], parameters=schema.model_json_schema())
        for name, schema in TOOL_SCHEMAS.items()
    ]


def run_task(task: Task, adapter: AgentAdapter, max_turns: int = 4) -> AgentRun:
    """Run an isolated task; one model turn may contain several calls.

    A new conversation and mock invocation context are built for every run. Tool
    observations include exact simulator output, never expected benchmark facts.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    messages = [
        Message(
            role=Role.SYSTEM,
            content=(
                "Use the provided sandbox tools for requested actions. "
                "Never claim an operation succeeded unless a tool confirms it. "
                "Explain the observed result in the user's language."
            ),
        ),
        Message(role=Role.USER, content=task.prompt),
    ]
    actions: list[ExecutedAction] = []
    specs = tool_specs()
    start = perf_counter()
    for _ in range(max_turns):
        try:
            turn = adapter.complete(list(messages), list(specs))
            if not isinstance(turn, ModelTurn):
                turn = ModelTurn.model_validate(turn)
        except Exception as exc:
            # Return only the exception class; exception text could contain API secrets.
            return AgentRun(
                task_id=task.id, messages=messages, actions=actions, final_answer="",
                termination=Termination.ADAPTER_ERROR,
                elapsed_ms=(perf_counter() - start) * 1000,
                error_type=type(exc).__name__,
            )
        messages.append(Message(
            role=Role.ASSISTANT, content=turn.content, tool_calls=turn.tool_calls
        ))
        if not turn.tool_calls:
            return AgentRun(
                task_id=task.id, messages=messages, actions=actions,
                final_answer=turn.content,
                termination=Termination.COMPLETE,
                elapsed_ms=(perf_counter() - start) * 1000,
            )
        for call in turn.tool_calls:
            result = execute_mock(call)
            actions.append(ExecutedAction(call=call, result=result))
            messages.append(Message(
                role=Role.TOOL,
                tool_name=call.name,
                tool_call_id=call.call_id,
                content=json.dumps(result.model_dump(), ensure_ascii=False),
            ))
    return AgentRun(
        task_id=task.id, messages=messages, actions=actions, final_answer="",
        termination=Termination.MAX_TURNS, elapsed_ms=(perf_counter() - start) * 1000,
    )
