"""Model runs must remain non-overwriting and research-label aware."""

import hashlib
import json
from pathlib import Path

import pytest

from sindhi_agentbench.adapters import AdapterConfig
from sindhi_agentbench.benchmark import run_benchmark
from sindhi_agentbench.dataset import load_tasks
from sindhi_agentbench.runner import ModelTurn

DATA = Path(__file__).resolve().parents[1] / "data" / "seed_tasks.jsonl"


class FakeAgent:
    config = AdapterConfig("ollama", "local-fake", temperature=0.0, seed=7)

    def __init__(self):
        self.turns = 0

    def complete(self, messages, tools):
        self.turns += 1
        if self.turns % 2:
            return ModelTurn(tool_calls=[load_tasks(DATA)[0].expected_call])
        return ModelTurn(content="Confirmed by simulated tool.")


def trace_state_has_expected_balance(directory):
    trace = json.loads((directory / "traces.jsonl").read_text().strip())
    return trace["environment_state"]["balances"]["A-100"] == 5000


def test_no_unverified_release_without_explicit_opt_in(tmp_path):
    with pytest.raises(ValueError, match="unverified"):
        run_benchmark(DATA, FakeAgent(), tmp_path / "output", max_tasks=1)
    assert not (tmp_path / "output").exists()


def test_smoke_run_writes_replayable_artifacts_and_never_overwrites(tmp_path):
    directory = tmp_path / "experiment"
    result = run_benchmark(
        DATA, FakeAgent(), directory, max_tasks=1, allow_unverified=True,
    )
    assert result["overall"]["task_completion"] == 1
    assert result["manifest"]["leaderboard_eligible"] is False
    assert result["manifest"]["max_tool_calls"] == 12
    assert result["manifest"]["termination_counts"]["max_tool_calls"] == 0
    assert trace_state_has_expected_balance(directory)
    assert result["manifest"]["dataset_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert json.loads((directory / "report.json").read_text())["evaluated_tasks"] == 1
    trace = json.loads((directory / "traces.jsonl").read_text().strip())
    assert trace["actions"][0]["result"]["status"] == "success"
    assert json.loads((directory / "manifest.json").read_text())["model_config"]["seed"] == 7
    with pytest.raises(FileExistsError):
        run_benchmark(DATA, FakeAgent(), directory, max_tasks=1, allow_unverified=True)


@pytest.mark.parametrize("invalid", [0, -1])
def test_invalid_limits_rejected(tmp_path, invalid):
    with pytest.raises(ValueError):
        run_benchmark(DATA, FakeAgent(), tmp_path / "a", max_tasks=invalid, allow_unverified=True)
    with pytest.raises(ValueError):
        run_benchmark(DATA, FakeAgent(), tmp_path / "b", max_turns=invalid, allow_unverified=True)
    with pytest.raises(ValueError):
        run_benchmark(
            DATA, FakeAgent(), tmp_path / "c",
            max_tool_calls=invalid, allow_unverified=True,
        )
