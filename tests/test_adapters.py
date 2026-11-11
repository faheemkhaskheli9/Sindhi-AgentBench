"""Offline SDK-contract tests: no credentials, payments, or live HTTP calls."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sindhi_agentbench.adapters import AdapterConfig, NativeAdapter
from sindhi_agentbench.dataset import load_tasks
from sindhi_agentbench.runner import Termination, run_task


TASK = load_tasks(Path(__file__).resolve().parents[1] / "data" / "seed_tasks.jsonl")[-1]


class Spy:
    def __init__(self, response, method):
        self.response = response
        self.calls = []
        self.method = method
        if method == "openai":
            self.chat = NS(completions=NS(create=self.invoke))
        elif method == "claude":
            self.messages = NS(create=self.invoke)
        elif method == "gemini":
            self.models = NS(generate_content=self.invoke)
        elif method == "ollama":
            self.chat = self.invoke
        elif method == "huggingface":
            self.chat_completion = self.invoke

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def openai_response():
    return {"choices": [{"message": {
        "content": None,
        "tool_calls": [{"id": "call-test", "function": {
            "name": "government_check_application",
            "arguments": '{"application_id":"APP-123"}',
        }}],
    }}]}


@pytest.mark.parametrize("provider", ["openai", "huggingface"])
def test_openai_compatible_adapters_roundtrip(provider):
    spy = Spy(openai_response(), provider)
    adapter = NativeAdapter(AdapterConfig(provider, "fake-model"), client=spy)
    first = adapter.complete(
        [__import__("sindhi_agentbench.runner", fromlist=["Message"]).Message(
            role="user", content=TASK.prompt
        )], __import__("sindhi_agentbench.runner", fromlist=["tool_specs"]).tool_specs()
    )
    assert first.tool_calls[0].name == "government.check_application"
    assert first.tool_calls[0].call_id == "call-test"
    assert spy.calls[0]["tools"][3]["function"]["name"] == "government_check_application"
    from sindhi_agentbench.runner import Message, Role, tool_specs
    second_messages = [
        Message(role=Role.USER, content=TASK.prompt),
        Message(role=Role.ASSISTANT, tool_calls=first.tool_calls),
        Message(role=Role.TOOL, content='{"status":"success"}', tool_name=first.tool_calls[0].name,
                tool_call_id="call-test"),
    ]
    adapter.complete(second_messages, tool_specs())
    sent = spy.calls[-1]["messages"]
    assert sent[-2]["tool_calls"][0]["id"] == "call-test"
    assert sent[-1]["tool_call_id"] == "call-test"


def test_claude_tool_result_linking_and_grouping():
    spy = Spy({"content": [{"type": "tool_use", "id": "toolu-1",
                            "name": "government_check_application",
                            "input": {"application_id": "APP-123"}}]}, "claude")
    adapter = NativeAdapter(AdapterConfig("claude", "fake"), client=spy)
    from sindhi_agentbench.runner import Message, Role, tool_specs
    turn = adapter.complete([Message(role=Role.USER, content=TASK.prompt)], tool_specs())
    call = turn.tool_calls[0]
    assert call.call_id == "toolu-1"
    adapter.complete([
        Message(role=Role.SYSTEM, content="sandbox"),
        Message(role=Role.USER, content=TASK.prompt),
        Message(role=Role.ASSISTANT, tool_calls=[call]),
        Message(role=Role.TOOL, tool_call_id="toolu-1",
                tool_name=call.name, content='{"status":"success"}'),
    ], tool_specs())
    history = spy.calls[-1]["messages"]
    assert history[-2]["content"][0]["id"] == "toolu-1"
    assert history[-1]["content"][0]["tool_use_id"] == "toolu-1"


def test_gemini_native_function_calls_and_observations():
    spy = Spy({"candidates": [{"content": {"parts": [
        {"function_call": {"name": "government_check_application",
                           "args": {"application_id": "APP-123"}, "id": "gem-1"}}
    ]}}]}, "gemini")
    adapter = NativeAdapter(AdapterConfig("gemini", "fake"), client=spy)
    from sindhi_agentbench.runner import Message, Role, tool_specs
    turn = adapter.complete([Message(role=Role.USER, content=TASK.prompt)], tool_specs())
    call = turn.tool_calls[0]
    assert call.call_id == "gem-1"
    adapter.complete([
        Message(role=Role.SYSTEM, content="sandbox"),
        Message(role=Role.USER, content=TASK.prompt),
        Message(role=Role.ASSISTANT, tool_calls=[call]),
        Message(role=Role.TOOL, tool_call_id="gem-1",
                tool_name=call.name, content='{"status":"success"}'),
    ], tool_specs())
    contents = spy.calls[-1]["contents"]
    assert contents[-2]["parts"][0]["function_call"]["name"] == "government_check_application"
    assert contents[-1]["parts"][0]["function_response"]["id"] == "gem-1"
    assert spy.calls[-1]["config"]["automatic_function_calling"]["disable"]


def test_ollama_native_call_object_arguments_and_observations():
    spy = Spy({"message": {"tool_calls": [{"function": {
        "name": "government_check_application",
        "arguments": {"application_id": "APP-123"},
    }}]}}, "ollama")
    adapter = NativeAdapter(AdapterConfig("ollama", "fake", seed=5), client=spy)
    from sindhi_agentbench.runner import Message, Role, tool_specs
    turn = adapter.complete([Message(role=Role.USER, content=TASK.prompt)], tool_specs())
    call = turn.tool_calls[0]
    assert call.name == "government.check_application"
    assert spy.calls[-1]["options"]["seed"] == 5
    adapter.complete([
        Message(role=Role.USER, content=TASK.prompt),
        Message(role=Role.ASSISTANT, tool_calls=[call]),
        Message(role=Role.TOOL, tool_name=call.name, content='{"status":"success"}'),
    ], tool_specs())
    assert spy.calls[-1]["messages"][-1]["role"] == "tool"


def test_invalid_arguments_are_penalized_not_crash():
    spy = Spy({"choices": [{"message": {"tool_calls": [{"id": "bad", "function": {
        "name": "government_check_application", "arguments": "NOT JSON"}}]}}]}, "openai")
    adapter = NativeAdapter(AdapterConfig("openai", "fake"), client=spy)
    run = run_task(TASK, adapter, max_turns=1)
    assert run.termination == Termination.MAX_TURNS
    assert run.actions[0].result.facts == {"error": "invalid_arguments"}


def test_seed_rejected_when_provider_cannot_honor_it():
    with pytest.raises(ValueError, match="Seed"):
        AdapterConfig("claude", "fake", seed=8)
