# Sindhi-AgentBench

**Goal:** Determine whether tool-using LLM agents can correctly complete real-world simulated tasks in **Sindhi**, **Roman Sindhi**, **Urdu**, and **English** — from intent comprehension and parameter extraction to API action and grounded explanation.

Unlike ordinary language-generation benchmarks, Sindhi-AgentBench measures what an agent **does**, not just what it says.

## Implemented (offline research harness)

- Strict Pydantic task, prediction, and independent-review schemas.
- Deterministic task-local **stateful mock** banking, booking, shopping, and
  government-service tools. Balances, seats and stock persist during a run;
  duplicate writes and unconfirmed transfers are rejected. No real-world
  transactions or external service calls.
- Tool selection, strict argument accuracy, and simulated completion scoring by language.
- Human-reviewed language fidelity and hallucination rates are `null` until an independent review is provided; missing reviews are never treated as perfect scores.
- Provider-neutral bounded multi-turn runner with mock tool responses, no live API credentials or real service effects.
- CLI, CI, unit tests and 16 **synthetic, unverified** starter cases (4 languages × 4 scenarios).
- 8 clarification/refusal fixtures, 4 explicit-confirmation fixtures and
  24 rejection/injection scenarios, all synthetic and unverified. The current
  fixture total is 52 tasks across four languages.

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

`task_completion` currently verifies simulated action success plus a nonempty final answer, **not** the semantic truthfulness of the answer. That requires independent review. Each **agent run** starts a fresh deterministic simulated session; tool calls
within a run share mutable state. A correctly rejected operation is not considered
a successful end-to-end answer without an independent human review.

## Run an actual model

Install only the provider you use. Examples:

```bash
pip install -e '.[dev,openai]'   # OPENAI_API_KEY environment variable required
pip install -e '.[gemini]'       # GOOGLE_API_KEY environment variable required
pip install -e '.[claude]'       # ANTHROPIC_API_KEY environment variable required
pip install -e '.[huggingface]'  # HF_TOKEN; remote inference provider must support tools
pip install -e '.[ollama]'       # Local Ollama server and downloaded model needed
```

Run an isolated tool simulation using a model (real model calls may incur charges):

```bash
sindhi-agentbench run-model data/seed_tasks.jsonl \
  --provider ollama --model YOUR_TOOL_CAPABLE_MODEL \
  --output-dir results/local-smoke-01 --max-tasks 4 \
  --max-tool-calls 12 --allow-unverified
```

The CLI writes a run manifest (including dataset SHA-256, provider SDK version, model
settings and task selection), normalized predictions, tool transcripts and scored report.
It **refuses to overwrite** a previous run. The starter dataset is unverified, so
`--allow-unverified` is required for smoke testing and the manifest always flags
runs as **not eligible** for a research leaderboard. Do not commit credentials or
personal information to traces. Different models/providers may not support the same
subset of tool JSON schemas; unsupported native tool calling should fail visibly.

Supported native tool adapters: OpenAI, Gemini, Claude, Hugging Face Inference Providers,
and Ollama. The Hugging Face adapter uses hosted inference rather than local Transformers.
No API-key-dependent tests run in CI; real-provider compatibility must be validated
before publishing baseline numbers.

## Integrate a model adapter

Implement the `AgentAdapter.complete(messages, tools) -> ModelTurn` interface from `sindhi_agentbench.runner`. A task runs through `run_task(task, adapter)` with a bounded tool-execution loop, and `run.as_prediction()` can be scored with the existing evaluator. Tool results are strictly fictional and isolated. Provider SDK integrations are tracked separately in GitHub issues.

## Research and development

See [docs/ROADMAP.md](docs/ROADMAP.md) for the prioritized backlog, annotation protocol, leakage prevention, model adapters, dashboard, and publication gates. The hard blocker for a credible research release is **100–200 independently human-verified, diverse multilingual agent tasks**.

All four fixture datasets are synthetic and **unverified**. They must not
be used to claim that any model outperforms another. Avoid placing private customer or government information in examples or logs.

## License

Code is MIT-licensed. Licensing and consent for the future human-annotated dataset must be established separately before publication.

## Clarification and refusal smoke tests

`data/edge_cases.jsonl` supplies 8 unverified examples where calling a tool is
incorrect. The evaluator can automatically check whether any tool was called, but
`task_completion` stays `null` for no-call responses until an independent reviewer
confirms that the user was properly asked for missing information or that a false
assertion was appropriately refused. Review `data/EDGE_CASES.md` for scope.

## Next P0 milestone: stateful transaction and failure evaluation

See [the phase plan](docs/plans/13-stateful-tool-failures.md). Run the four
fixture validators to check gold calls against the simulator:

```bash
for file in data/*.jsonl; do sindhi-agentbench validate "$file"; done
```

Each run stores the final fictional session state in the trace. Repeated
mutating tool calls are rejected; prompt-injection notes returned by the
government mock are untrusted observations, not instructions. For expected
service failures, the automatic `execution_outcome` metric checks whether
the simulator produced the correct rejection. `task_completion` remains
`null` until an independent human verifies the agent's explanation.

**Research status:** 52 synthetic fixtures; 0 human-verified tasks; no
published model scores. All provider integrations still need live-model
compatibility checks.
