from pathlib import Path

import pytest
from pydantic import ValidationError

from sindhi_agentbench.dataset import load_tasks, read_jsonl, unique_by_task_id, validate_tasks
from sindhi_agentbench.schema import Prediction, Task

DATASET = Path(__file__).resolve().parents[1] / "data" / "seed_tasks.jsonl"


def test_seed_tasks_validate():
    tasks = load_tasks(DATASET)
    assert len(tasks) == 16
    assert {t.language.value for t in tasks} == {"sd", "sd-Latn", "ur", "en"}
    assert {t.domain.value for t in tasks} == {"banking", "booking", "shopping", "government"}
    assert all(t.verification_status == "unverified" for t in tasks)


def test_verified_task_needs_independent_human_review():
    payload = load_tasks(DATASET)[0].model_dump()
    payload["verification_status"] = "verified"
    with pytest.raises(ValidationError):
        Task.model_validate(payload)
    payload["annotator_id"] = "annotator"
    payload["reviewer_id"] = "annotator"
    with pytest.raises(ValidationError):
        Task.model_validate(payload)
    payload["reviewer_id"] = "second-person"
    assert Task.model_validate(payload).verification_status == "verified"


def test_duplicate_tasks_fail():
    task = load_tasks(DATASET)[0]
    with pytest.raises(ValueError, match="Duplicate"):
        unique_by_task_id([task, task])


def test_gold_result_disagreement_fails():
    task = load_tasks(DATASET)[0].model_copy(update={"expected_facts": {"amount_pkr": 999}})
    with pytest.raises(ValueError, match="contradicts"):
        validate_tasks([task])


@pytest.mark.parametrize(
    "payload, key",
    [
        ('{"task_id":"a","task_id":"b"}', "task_id"),
        (
            '{"task_id":"a","tool_call":{"name":"bank.transfer",'
            '"arguments":{"amount_pkr":1,"amount_pkr":2}}}',
            "amount_pkr",
        ),
    ],
)
def test_jsonl_rejects_duplicate_keys_at_any_depth(tmp_path, payload, key):
    path = tmp_path / "predictions.jsonl"
    path.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"predictions.jsonl:1: Duplicate JSON key: {key}"):
        read_jsonl(path, Prediction)


def test_jsonl_allows_same_key_in_separate_objects(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"task_id":"a","tool_call":{"name":"bank.transfer","arguments":{"x":1}},'
        '"tool_calls":[{"name":"bank.transfer","arguments":{"x":1}}]}\n',
        encoding="utf-8",
    )
    assert read_jsonl(path, Prediction)[0].task_id == "a"
