from pathlib import Path

import pytest

from sindhi_agentbench.dataset import load_tasks
from sindhi_agentbench.evaluate import arguments_match, evaluate, score_task
from sindhi_agentbench.schema import HumanReview, Prediction, ToolCall

DATASET = Path(__file__).resolve().parents[1] / "data" / "seed_tasks.jsonl"


def predictions_for(tasks):
    return {t.id: Prediction(
        task_id=t.id, tool_call=t.expected_call, final_answer="Test response"
    ) for t in tasks}


def test_oracle_trace_has_perfect_structured_metrics_but_unscored_human_metrics():
    tasks = load_tasks(DATASET)
    report = evaluate(tasks, predictions_for(tasks))
    assert report.overall["tool_selection"] == 1
    assert report.overall["parameter_accuracy"] == 1
    assert report.overall["task_completion"] == 1
    assert report.overall["language_fidelity"] is None
    assert report.overall["hallucination_rate"] is None
    assert report.overall["hallucination_rate_reviewed_count"] == 0
    assert len(report.by_language) == 4


def test_mismatched_arguments_and_wrong_tool_do_not_complete_task():
    task = load_tasks(DATASET)[0]
    predictions = predictions_for([task])
    prediction = predictions[task.id]
    wrong = prediction.model_copy(update={"tool_call": ToolCall(
        name="bank.transfer", arguments={
            "from_account": "A-100", "to_account": "B-200", "amount_pkr": 300
        }
    )})
    result = score_task(task, wrong)
    assert result.tool_selection == 1
    assert result.parameter_accuracy < 1
    assert result.task_completion == 0
    wrong_tool = prediction.model_copy(update={"tool_call": ToolCall(
        name="government.check_application", arguments={"application_id": "APP-123"}
    )})
    result = score_task(task, wrong_tool)
    assert result.tool_selection == 0
    assert result.parameter_accuracy == 0
    assert result.task_completion == 0


def test_extra_arguments_and_bool_int_coercion_are_penalized():
    expected = ToolCall(name="bank.transfer", arguments={"amount_pkr": 1})
    assert arguments_match(expected, ToolCall(
        name="bank.transfer", arguments={"amount_pkr": True}
    )) == 0
    assert arguments_match(expected, ToolCall(
        name="bank.transfer", arguments={"amount_pkr": 1, "extra": "x"}
    )) == 0.5


def test_reviews_are_independent_inputs():
    task = load_tasks(DATASET)[0]
    reviewed = HumanReview(task_id=task.id, reviewer_id="independent-reviewer",
                           language_fidelity=0.7, hallucinated=True)
    report = evaluate([task], predictions_for([task]), {task.id: reviewed})
    assert report.overall["language_fidelity"] == 0.7
    assert report.overall["hallucination_rate"] == 1


def test_missing_or_unknown_prediction_fails():
    tasks = load_tasks(DATASET)
    with pytest.raises(ValueError, match="Missing predictions"):
        evaluate(tasks, {})
    predictions = predictions_for(tasks)
    predictions["not-a-task"] = Prediction(task_id="not-a-task")
    with pytest.raises(ValueError, match="Unknown task ids"):
        evaluate(tasks, predictions)
