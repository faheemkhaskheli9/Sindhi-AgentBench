"""Native function-calling adapters. SDKs are imported only when selected.

All APIs receive only the user prompt, conversation, and public tool schemas.
Gold calls and expected facts are never exposed to providers.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

from sindhi_agentbench.runner import Message, ModelTurn, Role, ToolSpec
from sindhi_agentbench.schema import ToolCall

Provider = Literal["openai", "gemini", "claude", "huggingface", "ollama"]


def _field(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _arguments(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError):
            pass
    # Invalid model output is an incorrect tool call, not an adapter exception.
    return {"__invalid_arguments__": True}


def _alias_map(tools: list[ToolSpec]) -> tuple[dict[str, str], dict[str, str]]:
    forward = {tool.name: tool.name.replace(".", "_") for tool in tools}
    if len(set(forward.values())) != len(forward):
        raise ValueError("Tool aliases collide")
    return forward, {alias: name for name, alias in forward.items()}


def _tool_declarations(tools: list[ToolSpec], forward: dict[str, str]) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": forward[tool.name],
            "description": tool.description,
            "parameters": tool.parameters,
        }}
        for tool in tools
    ]


def _ids(messages: list[Message]) -> dict[tuple[int, int | None], str]:
    ids: dict[tuple[int, int | None], str] = {}
    pending: list[str] = []
    for i, msg in enumerate(messages):
        if msg.role == Role.ASSISTANT:
            for j, call in enumerate(msg.tool_calls):
                value = call.call_id or f"call_{i}_{j}"
                ids[(i, j)] = value
                pending.append(value)
        if msg.role == Role.TOOL:
            if msg.tool_call_id:
                ids[(i, None)] = msg.tool_call_id
                if msg.tool_call_id in pending:
                    pending.remove(msg.tool_call_id)
            else:
                ids[(i, None)] = pending.pop(0) if pending else f"unmatched_{i}"
    return ids


def _openai_messages(messages: list[Message], forward: dict[str, str]) -> list[dict]:
    ids = _ids(messages)
    output: list[dict] = []
    for i, message in enumerate(messages):
        if message.role == Role.TOOL:
            output.append({
                "role": "tool", "content": message.content,
                "tool_call_id": ids[(i, None)],
            })
        elif message.role == Role.ASSISTANT:
            row: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                row["tool_calls"] = [
                    {"id": ids[(i, j)], "type": "function", "function": {
                        "name": forward.get(call.name, call.name),
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    }}
                    for j, call in enumerate(message.tool_calls)
                ]
            output.append(row)
        else:
            output.append({"role": message.role.value, "content": message.content})
    return output


def _openai_turn(response: Any, reverse: dict[str, str]) -> ModelTurn:
    message = _field(_field(response, "choices")[0], "message")
    calls = []
    for entry in _field(message, "tool_calls") or []:
        function = _field(entry, "function")
        calls.append(ToolCall(
            name=reverse.get(_field(function, "name"), _field(function, "name")),
            arguments=_arguments(_field(function, "arguments")),
            call_id=_field(entry, "id"),
        ))
    return ModelTurn(content=_field(message, "content") or "", tool_calls=calls)


def _claude_messages(messages: list[Message], forward: dict[str, str]) -> list[dict]:
    ids = _ids(messages)
    output: list[dict] = []
    for i, message in enumerate(messages):
        if message.role == Role.SYSTEM:
            continue
        if message.role == Role.TOOL:
            block = {
                "type": "tool_result", "tool_use_id": ids[(i, None)],
                "content": message.content,
            }
            if output and output[-1].get("_tool_results"):
                output[-1]["content"].append(block)
            else:
                output.append({"role": "user", "content": [block], "_tool_results": True})
        elif message.role == Role.ASSISTANT:
            blocks: list[dict] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for j, call in enumerate(message.tool_calls):
                blocks.append({
                    "type": "tool_use", "id": ids[(i, j)],
                    "name": forward.get(call.name, call.name),
                    "input": call.arguments,
                })
            if blocks:
                output.append({"role": "assistant", "content": blocks})
        else:
            output.append({"role": "user", "content": message.content})
    for row in output:
        row.pop("_tool_results", None)
    return output


def _gemini_schema(schema: dict) -> dict:
    """Keep the supported JSON-schema subset; omit Pydantic-only keywords."""
    allowed = {"type", "description", "enum", "properties", "items", "required", "format"}
    return {
        key: (
            {k: _gemini_schema(v) for k, v in value.items()}
            if key == "properties" else
            _gemini_schema(value) if key == "items" else value
        )
        for key, value in schema.items() if key in allowed
    }


def _gemini_contents(messages: list[Message], forward: dict[str, str]) -> list[dict]:
    ids = _ids(messages)
    output: list[dict] = []
    for i, message in enumerate(messages):
        if message.role == Role.SYSTEM:
            continue
        if message.role == Role.TOOL:
            response = {
                "name": forward.get(message.tool_name, message.tool_name),
                "response": json.loads(message.content),
            }
            if message.tool_call_id:
                response["id"] = message.tool_call_id
            part = {"function_response": response}
            if output and output[-1].get("_tool_results"):
                output[-1]["parts"].append(part)
            else:
                output.append({"role": "user", "parts": [part], "_tool_results": True})
        elif message.role == Role.ASSISTANT:
            parts: list[dict] = []
            if message.content:
                parts.append({"text": message.content})
            for j, call in enumerate(message.tool_calls):
                part: dict[str, Any] = {
                    "name": forward.get(call.name, call.name), "args": call.arguments
                }
                if call.call_id:
                    part["id"] = ids[(i, j)]
                parts.append({"function_call": part})
            if parts:
                output.append({"role": "model", "parts": parts})
        else:
            output.append({"role": "user", "parts": [{"text": message.content}]})
    for row in output:
        row.pop("_tool_results", None)
    return output


def _ollama_messages(messages: list[Message], forward: dict[str, str]) -> list[dict]:
    output: list[dict] = []
    for message in messages:
        if message.role == Role.TOOL:
            output.append({
                "role": "tool", "tool_name": forward.get(message.tool_name, message.tool_name),
                "content": message.content,
            })
        elif message.role == Role.ASSISTANT:
            row: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                row["tool_calls"] = [
                    {"type": "function", "function": {
                        "name": forward.get(call.name, call.name),
                        "arguments": call.arguments,
                    }}
                    for call in message.tool_calls
                ]
            output.append(row)
        else:
            output.append({"role": message.role.value, "content": message.content})
    return output


@dataclass(frozen=True)
class AdapterConfig:
    provider: Provider
    model: str
    temperature: float | None = None
    max_tokens: int = 512
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.provider not in ("openai", "gemini", "claude", "huggingface", "ollama"):
            raise ValueError("Unknown provider")
        if not self.model.strip():
            raise ValueError("Model is required")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be within 0..2")
        if self.seed is not None and self.provider not in ("openai", "ollama"):
            raise ValueError("Seed support is not guaranteed for this provider")


class NativeAdapter:
    """Tool-calling model adapter; injectable clients enable offline contract tests.

    Providers differ in schema support and model-level tool support. An unsupported
    model should fail rather than silently switch to prompted JSON mode.
    """

    def __init__(self, config: AdapterConfig, client: Any = None):
        self.config = config
        self.client = client if client is not None else self._make_client()

    def _make_client(self) -> Any:
        match self.config.provider:
            case "openai":
                from openai import OpenAI
                return OpenAI()
            case "huggingface":
                from huggingface_hub import InferenceClient
                return InferenceClient()
            case "claude":
                from anthropic import Anthropic
                return Anthropic()
            case "gemini":
                from google import genai
                return genai.Client()
            case "ollama":
                from ollama import Client
                return Client()
            case _:
                raise AssertionError("Unreachable provider")

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        cfg = self.config
        forward, reverse = _alias_map(tools)
        declarations = _tool_declarations(tools, forward)
        if cfg.provider in ("openai", "huggingface"):
            kwargs: dict[str, Any] = {
                "model": cfg.model, "messages": _openai_messages(messages, forward),
                "tools": declarations, "tool_choice": "auto",
            }
            if cfg.provider == "openai":
                kwargs["max_completion_tokens"] = cfg.max_tokens
                if cfg.seed is not None:
                    kwargs["seed"] = cfg.seed
            else:
                kwargs["max_tokens"] = cfg.max_tokens
            if cfg.temperature is not None:
                kwargs["temperature"] = cfg.temperature
            if cfg.provider == "openai":
                response = self.client.chat.completions.create(**kwargs)
            else:
                response = self.client.chat_completion(**kwargs)
            return _openai_turn(response, reverse)
        if cfg.provider == "claude":
            kwargs = {
                "model": cfg.model, "max_tokens": cfg.max_tokens,
                "system": "\n".join(msg.content for msg in messages if msg.role == Role.SYSTEM),
                "messages": _claude_messages(messages, forward),
                "tools": [
                    {"name": forward[tool.name], "description": tool.description,
                     "input_schema": tool.parameters} for tool in tools
                ],
            }
            if cfg.temperature is not None:
                kwargs["temperature"] = cfg.temperature
            response = self.client.messages.create(**kwargs)
            calls = []
            text_parts = []
            for block in _field(response, "content") or []:
                if _field(block, "type") == "tool_use":
                    name = _field(block, "name")
                    calls.append(ToolCall(
                        name=reverse.get(name, name),
                        arguments=_arguments(_field(block, "input")),
                        call_id=_field(block, "id"),
                    ))
                elif _field(block, "type") == "text":
                    text_parts.append(_field(block, "text") or "")
            return ModelTurn(content="\n".join(text_parts), tool_calls=calls)
        if cfg.provider == "gemini":
            generation: dict[str, Any] = {
                "tools": [{"function_declarations": [
                    {"name": forward[tool.name], "description": tool.description,
                     "parameters": _gemini_schema(tool.parameters)}
                    for tool in tools
                ]}],
                "automatic_function_calling": {"disable": True},
                "max_output_tokens": cfg.max_tokens,
                "system_instruction": "\n".join(
                    msg.content for msg in messages if msg.role == Role.SYSTEM
                ),
            }
            if cfg.temperature is not None:
                generation["temperature"] = cfg.temperature
            response = self.client.models.generate_content(
                model=cfg.model, contents=_gemini_contents(messages, forward),
                config=generation,
            )
            candidates = _field(response, "candidates") or []
            parts = _field(_field(candidates[0], "content"), "parts") or [] if candidates else []
            calls = []
            text_parts = []
            for part in parts:
                function = _field(part, "function_call")
                if function:
                    name = _field(function, "name")
                    calls.append(ToolCall(
                        name=reverse.get(name, name),
                        arguments=_arguments(_field(function, "args")),
                        call_id=_field(function, "id"),
                    ))
                elif _field(part, "text"):
                    text_parts.append(_field(part, "text"))
            return ModelTurn(content="\n".join(text_parts), tool_calls=calls)
        if cfg.provider == "ollama":
            options = {"num_predict": cfg.max_tokens}
            if cfg.temperature is not None:
                options["temperature"] = cfg.temperature
            if cfg.seed is not None:
                options["seed"] = cfg.seed
            response = self.client.chat(
                model=cfg.model, messages=_ollama_messages(messages, forward),
                tools=declarations, options=options, stream=False,
            )
            message = _field(response, "message")
            calls = []
            for entry in _field(message, "tool_calls") or []:
                function = _field(entry, "function")
                name = _field(function, "name")
                calls.append(ToolCall(
                    name=reverse.get(name, name),
                    arguments=_arguments(_field(function, "arguments")),
                ))
            return ModelTurn(content=_field(message, "content") or "", tool_calls=calls)
        raise AssertionError("Unreachable provider")
