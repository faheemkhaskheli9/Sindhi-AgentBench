"""Reproducible offline-simulator experiment artifacts for native model adapters."""

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sindhi_agentbench.adapters import NativeAdapter
from sindhi_agentbench.dataset import load_tasks
from sindhi_agentbench.evaluate import evaluate
from sindhi_agentbench.runner import run_task

SDK_PACKAGES = {
    "openai": "openai",
    "gemini": "google-genai",
    "claude": "anthropic",
    "huggingface": "huggingface-hub",
    "ollama": "ollama",
}


def _sdk_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def run_benchmark(
    dataset_path: Path,
    adapter: NativeAdapter,
    output_dir: Path,
    *,
    max_turns: int = 4,
    max_tasks: int | None = None,
    allow_unverified: bool = False,
) -> dict:
    """Write new, non-overwriting JSONL traces, predictions, and score report.

    The simulator is offline, but the model adapter may call a paid external API.
    Synthetic or unreviewed datasets are never marked leaderboard eligible.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    tasks = load_tasks(dataset_path)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if not allow_unverified and any(t.verification_status != "verified" for t in tasks):
        raise ValueError(
            "Dataset contains unverified tasks. Use --allow-unverified for smoke tests."
        )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    runs = [run_task(task, adapter, max_turns=max_turns) for task in tasks]
    predictions = {run.task_id: run.as_prediction() for run in runs}
    report = evaluate(tasks, predictions)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    cfg = adapter.config
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": digest,
        "dataset_filename": dataset_path.name,
        "dataset_task_count": len(load_tasks(dataset_path)),
        "selected_task_ids": [t.id for t in tasks],
        "verified_tasks": sum(t.verification_status == "verified" for t in tasks),
        "unverified_tasks": sum(t.verification_status != "verified" for t in tasks),
        "leaderboard_eligible": False,
        "leaderboard_note": "Requires reviewed held-out dataset and independent answer audits.",
        "tool_mode": "native_function_calling",
        "model_config": asdict(cfg),
        "sdk_version": _sdk_version(SDK_PACKAGES[cfg.provider]),
        "python_version": platform.python_version(),
        "max_turns": max_turns,
        "termination_counts": {
            reason: sum(run.termination.value == reason for run in runs)
            for reason in ("complete", "max_turns", "adapter_error")
        },
    }
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p.model_dump(), ensure_ascii=False) + "\n"
                for p in predictions.values()),
        encoding="utf-8",
    )
    (output_dir / "traces.jsonl").write_text(
        "".join(run.model_dump_json(exclude_none=True) + "\n" for run in runs),
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "overall": report.overall, "by_language": report.by_language}
