from pathlib import Path

import pytest
from pydantic import ValidationError

from sindhi_agentbench.dataset import load_tasks, unique_by_task_id, validate_tasks
from sindhi_agentbench.schema import Task

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
