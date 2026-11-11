"""Safe clarification/refusal examples and human-review gates."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sindhi_agentbench.dataset import load_tasks
from sindhi_agentbench.evaluate import evaluate, score_task
from sindhi_agentbench.schema import HumanReview, Prediction, Task, ToolCall

EDGE = Path(__file__).resolve().parents[1] / "data" / "edge_cases.jsonl"


def test_edge_cases_are_unverified_and_cover_two_safe_behaviors():
    tasks = load_tasks(EDGE)
    assert len(tasks) == 8
    assert {t.language.value for t in tasks} == {"sd", "sd-Latn", "ur", "en"}
    assert {t.expected_behavior for t in tasks} == {"clarify", "refuse"}
    assert all(t.expected_call is None for t in tasks)
    assert all(t.verification_status == "unverified" for t in tasks)


def test_no_tool_credit_requires_human_judgment_of_behavior():
    task = load_tasks(EDGE)[0]
    p = Prediction(task_id=task.id, final_answer="Which recipient account?")
    score = score_task(task, p)
    assert score.tool_selection == 1
    assert score.parameter_accuracy is None
    assert score.task_completion is None
    reviewer = HumanReview(
        task_id=task.id, reviewer_id="human-2", language_fidelity=1,
        hallucinated=False, behavior_correct=True,
    )
    score = score_task(task, p, reviewer)
    assert score.task_completion == 1
    assert score.behavior_correct == 1


def test_single_or_multiple_unauthorized_tool_calls_are_not_hidden():
    task = load_tasks(EDGE)[0]
    bad_call = ToolCall(name="bank.transfer", arguments={
        "from_account": "A-100", "to_account": "B-200", "amount_pkr": 500,
    })
    for calls in ([bad_call], [bad_call, bad_call]):
        p = Prediction(task_id=task.id, tool_calls=calls, final_answer="Done.")
        score = score_task(task, p)
        assert score.tool_selection == 0
        assert score.task_completion == 0


def test_no_action_validation_rejects_gold_tool_or_expected_facts():
    sample = load_tasks(EDGE)[0].model_dump()
    sample["expected_facts"] = {"reference": "fake"}
    with pytest.raises(ValidationError):
        Task.model_validate(sample)
    sample["expected_facts"] = {}
    sample["expected_call"] = {"name": "bank.transfer", "arguments": {}}
    with pytest.raises(ValidationError):
        Task.model_validate(sample)


def test_prediction_detects_contradictory_tool_lists():
    task = load_tasks(EDGE)[0]
    call = ToolCall(name="government.check_application", arguments={"application_id": "APP-123"})
    with pytest.raises(ValidationError):
        Prediction(task_id=task.id, tool_call=call, tool_calls=[call, call])


def test_no_call_summaries_preserve_nullable_metrics():
    tasks = load_tasks(EDGE)
    predictions = {t.id: Prediction(task_id=t.id, final_answer="Need information")
                   for t in tasks}
    report = evaluate(tasks, predictions)
    assert report.overall["tool_selection"] == 1
    assert report.overall["parameter_accuracy"] is None
    assert report.overall["task_completion"] is None
    assert report.overall["task_completion_reviewed_count"] == 0
