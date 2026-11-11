"""Dataset reading and structural validation. Review status is never inferred."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from sindhi_agentbench.environments import TOOL_SCHEMAS, execute_mock
from sindhi_agentbench.schema import Domain, HumanReview, Prediction, Task

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path, schema: type[T]) -> list[T]:
    rows: list[T] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(schema.model_validate(json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def unique_by_task_id(rows: list[T]) -> dict[str, T]:
    indexed: dict[str, T] = {}
    for row in rows:
        task_id = getattr(row, "task_id", getattr(row, "id", None))
        if task_id in indexed:
            raise ValueError(f"Duplicate task id: {task_id}")
        indexed[task_id] = row
    return indexed


def validate_tasks(tasks: list[Task]) -> None:
    unique_by_task_id(tasks)
    domain_tools = {
        Domain.BANKING: "bank.transfer",
        Domain.BOOKING: "booking.reserve",
        Domain.SHOPPING: "shopping.place_order",
        Domain.GOVERNMENT: "government.check_application",
    }
    if not tasks:
        raise ValueError("Dataset is empty")
    for task in tasks:
        if task.expected_behavior != "execute":
            continue
        assert task.expected_call is not None
        if task.expected_call.name not in TOOL_SCHEMAS:
            raise ValueError(f"{task.id}: unknown expected tool")
        if task.expected_call.name != domain_tools[task.domain]:
            raise ValueError(f"{task.id}: tool does not match domain")
        actual = execute_mock(task.expected_call)
        if actual.status != task.expected_status:
            raise ValueError(
                f"{task.id}: expected status {task.expected_status}, "
                f"simulator returned {actual.status}: {actual.facts}"
            )
        for key, value in task.expected_facts.items():
            if (
                key not in actual.facts
                or type(actual.facts[key]) is not type(value)
                or actual.facts[key] != value
            ):
                raise ValueError(f"{task.id}: expected fact {key} contradicts simulator")


def load_tasks(path: str | Path) -> list[Task]:
    tasks = read_jsonl(path, Task)
    validate_tasks(tasks)
    return tasks


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    return unique_by_task_id(read_jsonl(path, Prediction))


def load_reviews(path: str | Path) -> dict[str, HumanReview]:
    return unique_by_task_id(read_jsonl(path, HumanReview))
