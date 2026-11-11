"""Synthetic negative outcomes and confirmation fixtures; never claim research validity."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sindhi_agentbench.dataset import load_tasks, validate_tasks
from sindhi_agentbench.evaluate import evaluate, score_task
from sindhi_agentbench.runner import ModelTurn, run_task
from sindhi_agentbench.schema import HumanReview, Prediction, Task, ToolCall

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "data" / "scenario_tasks.jsonl"
CONFIRM = ROOT / "data" / "confirmation_tasks.jsonl"


class Scripted:
    def __init__(self, *turns):
        self.turns = iter(turns)

    def complete(self, messages, tools):
        return next(self.turns)


def test_scenarios_are_explicitly_unverified_with_real_mock_results():
    tasks = load_tasks(SCENARIOS)
    assert len(tasks) == 24
    assert {x.language.value for x in tasks} == {"sd", "sd-Latn", "ur", "en"}
    assert sum(x.expected_status == "rejected" for x in tasks) == 20
    assert all(x.verification_status == "unverified" for x in tasks)
    assert any(x.expected_call.arguments.get("application_id") == "APP-INJECT"
               for x in tasks)


def test_rejected_tool_outcome_is_not_automatically_completed():
    task = next(x for x in load_tasks(SCENARIOS) if x.expected_status == "rejected")
    guess = Prediction(task_id=task.id, tool_call=task.expected_call,
                       final_answer="The operation was rejected.")
    score = score_task(task, guess)
    assert score.tool_selection == 1
    assert score.parameter_accuracy == 1
    assert score.execution_outcome == 1
    assert score.task_completion is None
    reviewed = HumanReview(task_id=task.id, reviewer_id="reviewer-02",
                           language_fidelity=1.0, hallucinated=False,
                           behavior_correct=True)
    assert score_task(task, guess, reviewed).task_completion == 1
    bad_review = reviewed.model_copy(update={"behavior_correct": False})
    assert score_task(task, guess, bad_review).task_completion == 0


def test_wrong_expected_outcome_or_wrong_arguments_fail():
    task = next(x for x in load_tasks(SCENARIOS) if x.expected_status == "rejected")
    prediction = Prediction(task_id=task.id, tool_call=ToolCall(
        name=task.expected_call.name, arguments={**task.expected_call.arguments,
                                                 "amount_pkr": 1500}),
        final_answer="Complete")
    score = score_task(task, prediction)
    assert score.execution_outcome == 0
    assert score.task_completion == 0
    wrong = task.model_copy(update={"expected_status": "success"})
    with pytest.raises(ValueError, match="expected status"):
        validate_tasks([wrong])


def test_negative_and_success_cases_have_distinct_review_denominators():
    tasks = load_tasks(SCENARIOS)
    guesses = {t.id: Prediction(task_id=t.id, tool_call=t.expected_call,
                                final_answer="A tool result was returned.") for t in tasks}
    report = evaluate(tasks, guesses)
    assert report.overall["execution_outcome"] == 1
    assert report.overall["execution_outcome_reviewed_count"] == 24
    assert report.overall["task_completion_reviewed_count"] == 4
    assert report.overall["hallucination_rate"] is None


def test_confirmation_dataset_and_guarded_tool_execution():
    tasks = load_tasks(CONFIRM)
    assert len(tasks) == 4
    assert all(t.confirmation_required and t.expected_behavior == "clarify" for t in tasks)
    for task in tasks:
        call = ToolCall(name="bank.transfer", arguments={
            "from_account": "A-100", "to_account": "B-200", "amount_pkr": 500,
        })
        run = run_task(task, Scripted(
            ModelTurn(tool_calls=[call]), ModelTurn(content="Confirm first."),
        ))
        assert run.actions[0].result.facts["error"] == "confirmation_required"
        assert run.environment_state["balances"]["A-100"] == 5000
        assert score_task(task, run.as_prediction()).tool_selection == 0


def test_no_action_cannot_claim_rejected_gold_status():
    task = load_tasks(CONFIRM)[0].model_dump()
    task["expected_status"] = "rejected"
    with pytest.raises(ValidationError):
        Task.model_validate(task)
