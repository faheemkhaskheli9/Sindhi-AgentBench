# Sindhi-AgentBench

**Goal:** Determine whether tool-using LLM agents can correctly complete real-world simulated tasks in **Sindhi**, **Roman Sindhi**, **Urdu**, and **English** — from intent comprehension and parameter extraction to API action and grounded explanation.

Unlike ordinary language-generation benchmarks, Sindhi-AgentBench measures what an agent **does**, not just what it says.

## Implemented (v0.1: offline harness)

- Strict Pydantic task, prediction, and independent-review schemas.
- Deterministic, isolated **mock** banking, booking, shopping, and government-service tools. No real transactions or external API calls.
- Tool selection, strict argument accuracy, and simulated completion scoring by language.
- Human-reviewed language fidelity and hallucination rates are `null` until an independent review is provided; missing reviews are never treated as perfect scores.
- Provider-neutral bounded multi-turn runner with mock tool responses, no live API credentials or real service effects.
- CLI, CI, unit tests and 16 **synthetic, unverified** starter cases (4 languages × 4 scenarios).

## Get started

Python 3.11+:

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
sindhi-agentbench validate data/seed_tasks.jsonl
```

Generate **oracle traces to test the pipeline only** (these are NOT model predictions or leaderboard scores):

```bash
sindhi-agentbench oracle data/seed_tasks.jsonl --output /tmp/oracle.jsonl
sindhi-agentbench evaluate data/seed_tasks.jsonl --predictions /tmp/oracle.jsonl --output /tmp/report.json
```

To benchmark a real model, supply JSONL predictions with one object per task:

```json
{"task_id":"seed-government-en-001","tool_call":{"name":"government.check_application","arguments":{"application_id":"APP-123"}},"final_answer":"The application is under review."}
```

The other 15 tasks must also have corresponding predictions for a full-dataset run. `--reviews reviews.jsonl` optionally accepts independently created records such as:

```json
{"task_id":"seed-government-en-001","reviewer_id":"reviewer-01","language_fidelity":1.0,"hallucinated":false}
```

`task_completion` currently verifies simulated action success plus a nonempty final answer, **not** the semantic truthfulness of the answer. That requires independent review. Each tool execution is an isolated deterministic simulation, not a persistent transaction system.

## Integrate a model adapter

Implement the `AgentAdapter.complete(messages, tools) -> ModelTurn` interface from `sindhi_agentbench.runner`. A task runs through `run_task(task, adapter)` with a bounded tool-execution loop, and `run.as_prediction()` can be scored with the existing evaluator. Tool results are strictly fictional and isolated. Provider SDK integrations are tracked separately in GitHub issues.

## Research and development

See [docs/ROADMAP.md](docs/ROADMAP.md) for the prioritized backlog, annotation protocol, leakage prevention, model adapters, dashboard, and publication gates. The hard blocker for a credible research release is **100–200 independently human-verified, diverse multilingual agent tasks**.

The seed set is intentionally small, templated, synthetic, and **unverified**; it must not be used to claim any model outperforms another. Avoid placing private customer or government information in examples or logs.

## License

Code is MIT-licensed. Licensing and consent for the future human-annotated dataset must be established separately before publication.
