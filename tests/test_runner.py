import json
from pathlib import Path

from sindhi_agentbench.dataset import load_tasks
from sindhi_agentbench.evaluate import score_task
from sindhi_agentbench.runner import ModelTurn, Role, Termination, run_task, tool_specs
from sindhi_agentbench.schema import ToolCall

DATASET = Path(__file__).resolve().parents[1] / "data" / "seed_tasks.jsonl"


class ScriptedAdapter:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.received = []

    def complete(self, messages, tools):
        self.received.append((messages, tools))
        return next(self.turns)


def test_realistic_two_turn_call_and_response():
    task = load_tasks(DATASET)[-1]  # government English
    adapter = ScriptedAdapter([
        ModelTurn(tool_calls=[task.expected_call]),
        ModelTurn(content="The status is under review."),
    ])
    run = run_task(task, adapter)
    assert run.termination == Termination.COMPLETE
    assert run.actions[0].result.facts == {"application_status": "under_review"}
    assert score_task(task, run.as_prediction()).task_completion == 1
    assert len(adapter.received) == 2
    first_messages, first_tools = adapter.received[0]
    assert task.prompt in first_messages[-1].content
    assert all("expected" not in x.content for x in first_messages)
    assert len(first_tools) == 4
    assert {spec.name for spec in first_tools} == {
        "bank.transfer", "booking.reserve", "shopping.place_order", "government.check_application"
    }
    second_messages, _ = adapter.received[1]
    assert json.loads(second_messages[-1].content)["facts"]["application_status"] == "under_review"
    assert second_messages[-1].role == Role.TOOL


def test_no_tool_call_cannot_complete_tool_task():
    task = load_tasks(DATASET)[-1]
    run = run_task(task, ScriptedAdapter([ModelTurn(content="Done")]))
    assert run.termination == Termination.COMPLETE
    assert score_task(task, run.as_prediction()).task_completion == 0


def test_multiple_calls_cannot_be_rewarded_as_single_correct_action():
    task = load_tasks(DATASET)[-1]
    run = run_task(task, ScriptedAdapter([
        ModelTurn(tool_calls=[task.expected_call, task.expected_call]),
        ModelTurn(content="Finished"),
    ]))
    assert len(run.actions) == 2
    assert run.as_prediction().tool_call is None
    assert score_task(task, run.as_prediction()).task_completion == 0


def test_wrong_parameters_are_rejected_by_real_mock_environment():
    task = load_tasks(DATASET)[-1]
    call = ToolCall(name="government.check_application", arguments={"application_id": "MISSING"})
    run = run_task(task, ScriptedAdapter([
        ModelTurn(tool_calls=[call]), ModelTurn(content="No such application")
    ]))
    assert run.actions[0].result.status == "rejected"
    assert score_task(task, run.as_prediction()).task_completion == 0


def test_max_turns_stops_unbounded_calls():
    task = load_tasks(DATASET)[-1]
    run = run_task(task, ScriptedAdapter([
        ModelTurn(tool_calls=[task.expected_call]),
        ModelTurn(tool_calls=[task.expected_call]),
    ]), max_turns=2)
    assert run.termination == Termination.MAX_TURNS
    assert len(run.actions) == 2
    assert run.as_prediction().final_answer == ""


def test_adapter_error_sanitizes_exception_text():
    task = load_tasks(DATASET)[-1]

    class ExplodingAdapter:
        def complete(self, messages, tools):
            raise RuntimeError("TOKEN=supersecret")

    run = run_task(task, ExplodingAdapter())
    assert run.termination == Termination.ADAPTER_ERROR
    assert run.error_type == "RuntimeError"
    assert "supersecret" not in run.model_dump_json()


def test_no_turns_is_invalid():
    task = load_tasks(DATASET)[-1]
    try:
        run_task(task, ScriptedAdapter([]), max_turns=0)
    except ValueError as exc:
        assert "max_turns" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_tool_spec_schema_is_exportable():
    assert all(spec.parameters["type"] == "object" for spec in tool_specs())
