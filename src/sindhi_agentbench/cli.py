"""Command-line dataset validation and offline prediction evaluation."""

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()
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
