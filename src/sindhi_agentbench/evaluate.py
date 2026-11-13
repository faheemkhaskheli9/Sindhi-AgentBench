"""Deterministic trace scoring with nullable, human-reviewed qualitative metrics."""

from statistics import mean

from sindhi_agentbench.environments import execute_mock
from sindhi_agentbench.schema import HumanReview, Prediction, Report, Task, TaskScore, ToolCall

_METRICS = (
    "tool_selection",
    "parameter_accuracy",
    "task_completion",
    "behavior_correct",
    "execution_outcome",
    "language_fidelity",
    "hallucination_rate",
)


def same_value(a: object, b: object) -> bool:
    """Prevent bool == 1 or int == float from passing a strict argument check."""
    return type(a) is type(b) and a == b


def arguments_match(expected: ToolCall, actual: ToolCall) -> float:
    if expected.name != actual.name:
        return 0.0
    expected_args = expected.arguments
    supplied_args = actual.arguments
    extras = supplied_args.keys() - expected_args.keys()
    denominator = len(expected_args) + len(extras)
    if denominator == 0:
        return 1.0
    correct = sum(
        key in supplied_args and same_value(value, supplied_args[key])
        for key, value in expected_args.items()
    )
    return correct / denominator


def score_task(task: Task, prediction: Prediction, review: HumanReview | None = None) -> TaskScore:
    if task.id != prediction.task_id or (review is not None and review.task_id != task.id):
        raise ValueError("Task, prediction and review IDs must agree")
    calls = prediction.tool_calls or (
        [prediction.tool_call] if prediction.tool_call is not None else []
    )
    # Multiple actions cannot be hidden by collapsing into a single-call trace.
    call = calls[0] if len(calls) == 1 else None
    if task.expected_behavior != "execute":
        safe_no_call = not calls
        behavior_ok = review.behavior_correct if review is not None else None
        if not safe_no_call or not prediction.final_answer.strip():
            completion = 0.0
        elif behavior_ok is None:
            completion = None
        else:
            completion = float(behavior_ok)
        return TaskScore(
            task_id=task.id,
            language=task.language,
            tool_selection=float(safe_no_call),
            parameter_accuracy=None,
            task_completion=completion,
            behavior_correct=float(behavior_ok) if behavior_ok is not None else None,
            language_fidelity=review.language_fidelity if review else None,
            hallucination_rate=float(review.hallucinated) if review else None,
            simulated_result=None,
        )
    assert task.expected_call is not None
    tool_correct = float(call is not None and call.name == task.expected_call.name)
    param_accuracy = arguments_match(task.expected_call, call) if call is not None else 0.0
    result = execute_mock(call) if call is not None else None
    facts_match = result is not None and all(
        key in result.facts and same_value(value, result.facts[key])
        for key, value in task.expected_facts.items()
    )
    outcome_correct = bool(
        tool_correct == 1
        and param_accuracy == 1
        and result is not None
        and result.status == task.expected_status
        and facts_match
    )
    if task.expected_status == "rejected":
        # A correctly rejected tool action is not proof the agent explained
        # the failure faithfully. It requires an independent answer review.
        if not outcome_correct or not prediction.final_answer.strip():
            completed: float | None = 0.0
        elif review is None or review.behavior_correct is None:
            completed = None
        else:
            completed = float(review.behavior_correct)
    else:
        completed = float(outcome_correct and bool(prediction.final_answer.strip()))
    return TaskScore(
        task_id=task.id,
        language=task.language,
        tool_selection=tool_correct,
        parameter_accuracy=param_accuracy,
        task_completion=completed,
        behavior_correct=float(review.behavior_correct) if (
            review and review.behavior_correct is not None
        ) else None,
        language_fidelity=review.language_fidelity if review else None,
        hallucination_rate=float(review.hallucinated) if review else None,
        simulated_result=result.model_dump() if result else None,
        execution_outcome=float(outcome_correct),
    )


def summarize(scores: list[TaskScore]) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"count": len(scores)}
    for metric in _METRICS:
        valid = [getattr(score, metric) for score in scores if getattr(score, metric) is not None]
        summary[metric] = mean(valid) if valid else None
        summary[f"{metric}_reviewed_count"] = len(valid)
    return summary


def evaluate(
    tasks: list[Task],
    predictions: dict[str, Prediction],
    reviews: dict[str, HumanReview] | None = None,
) -> Report:
    """Evaluate a unique set of tasks without inflating denominators.

    Duplicate task IDs must fail even if predictions were keyed correctly:

    >>> from sindhi_agentbench.schema import ToolCall
    >>> task = Task(id="repeat", language="en", domain="government",
    ...             prompt="Check application",
    ...             expected_call=ToolCall(name="government.check_application",
    ...                                    arguments={"application_id": "APP-123"}))
    >>> guess = Prediction(task_id="repeat", tool_call=task.expected_call,
    ...                    final_answer="Checked")
    >>> evaluate([task, task], {"repeat": guess})
    Traceback (most recent call last):
    ...
    ValueError: Duplicate task ids: ['repeat']
    """
    reviews = reviews or {}
    ids = {task.id for task in tasks}
    if len(ids) != len(tasks):
        from collections import Counter

        counts = Counter(task.id for task in tasks)
        duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
        raise ValueError(f"Duplicate task ids: {duplicates}")
    unknown_predictions = predictions.keys() - ids
    unknown_reviews = reviews.keys() - ids
    if unknown_predictions or unknown_reviews:
        raise ValueError(f"Unknown task ids: {sorted(unknown_predictions | unknown_reviews)}")
    missing = ids - predictions.keys()
    if missing:
        raise ValueError(f"Missing predictions for tasks: {sorted(missing)}")
    scores = [score_task(task, predictions[task.id], reviews.get(task.id)) for task in tasks]
    languages = sorted({task.language.value for task in tasks})
    by_language = {
        lang: summarize([s for s in scores if s.language.value == lang])
        for lang in languages
    }
    return Report(
        evaluated_tasks=len(scores),
        by_language=by_language,
        overall=summarize(scores),
        scores=scores,
    )
