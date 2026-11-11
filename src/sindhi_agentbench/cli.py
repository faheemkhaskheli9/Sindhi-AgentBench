"""Command-line dataset validation and offline prediction evaluation."""

import argparse
import json
from pathlib import Path

from sindhi_agentbench.adapters import AdapterConfig, NativeAdapter
from sindhi_agentbench.benchmark import run_benchmark
from sindhi_agentbench.dataset import load_predictions, load_reviews, load_tasks
from sindhi_agentbench.evaluate import evaluate
from sindhi_agentbench.schema import Prediction


def main() -> None:
    parser = argparse.ArgumentParser(prog="sindhi-agentbench")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate dataset + gold mock calls")
    validate.add_argument("dataset", type=Path)
    oracle = commands.add_parser(
        "oracle", help="Generate synthetic gold traces for a harness smoke test"
    )
    oracle.add_argument("dataset", type=Path)
    oracle.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("evaluate", help="Score externally generated predictions")
    run.add_argument("dataset", type=Path)
    run.add_argument("--predictions", required=True, type=Path)
    run.add_argument("--reviews", type=Path, help="Independent human assessments (optional)")
    run.add_argument("--output", type=Path, help="Write detailed JSON report")
    model_run = commands.add_parser(
        "run-model", help="Run a model on isolated mock tools; SDK/API costs may apply"
    )
    model_run.add_argument("dataset", type=Path)
    model_run.add_argument("--provider", required=True, choices=[
        "openai", "gemini", "claude", "huggingface", "ollama"
    ])
    model_run.add_argument("--model", required=True)
    model_run.add_argument("--output-dir", type=Path, required=True)
    model_run.add_argument("--temperature", type=float)
    model_run.add_argument("--max-tokens", type=int, default=512)
    model_run.add_argument("--seed", type=int)
    model_run.add_argument("--max-turns", type=int, default=4)
    model_run.add_argument("--max-tasks", type=int)
    model_run.add_argument(
        "--allow-unverified", action="store_true",
        help="Permit unverified examples for smoke testing, not leaderboard claims"
    )
    args = parser.parse_args()
    if args.command == "run-model":
        config = AdapterConfig(
            provider=args.provider, model=args.model, temperature=args.temperature,
            max_tokens=args.max_tokens, seed=args.seed
        )
        result = run_benchmark(
            args.dataset, NativeAdapter(config), args.output_dir,
            max_turns=args.max_turns, max_tasks=args.max_tasks,
            allow_unverified=args.allow_unverified,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    tasks = load_tasks(args.dataset)
    if args.command == "validate":
        print(json.dumps({"valid_tasks": len(tasks), "verified": sum(
            t.verification_status == "verified" for t in tasks
        )}))
    elif args.command == "oracle":
        traces = [Prediction(
            task_id=t.id, tool_call=t.expected_call, final_answer="MOCK: see recorded tool result"
        ) for t in tasks]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row.model_dump(), ensure_ascii=False) + "\n" for row in traces),
            encoding="utf-8",
        )
        print("Oracle smoke-test traces only; do not rank these as model results.")
    else:
        report = evaluate(
            tasks,
            load_predictions(args.predictions),
            load_reviews(args.reviews) if args.reviews else None,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"overall": report.overall, "by_language": report.by_language}, indent=2))


if __name__ == "__main__":
    main()
