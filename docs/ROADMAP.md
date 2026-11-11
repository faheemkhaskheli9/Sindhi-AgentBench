# Development plan and release gates

## Goal

Evaluate **real task execution**, not generic fluency or multiple-choice questions. A model must interpret a Sindhi, Roman Sindhi, Urdu, or English request; select the correct tool; supply correct arguments; execute against isolated mock services; and accurately explain the verified outcome.

## P0 — Reproducible harness (initial implementation)

- [x] Strict versioned JSONL contracts for tasks, prediction traces, and separate human reviews.
- [x] Isolated deterministic tool simulators for banking, booking, shopping, and government services.
- [x] Offline tool-selection, exact-argument, and mock-execution scoring, stratified by language.
- [x] Synthetic starter tasks, unit tests, CLI, and GitHub Actions CI.
- [x] Provider-neutral multi-turn orchestration over task-local state with bounded
  turns and bounded tool-call bursts; five adapters implemented.
- [x] Add synthetic refusal and ambiguous-intent no-action tasks with call-count safety scoring.
- [x] Add synthetic code-switching, untrusted tool-output injection and
  failed-service scenarios (not yet native-speaker verified).
- [x] Add deterministic balance/stock/seat mutation, duplicate-call protection,
  confirmation gates, replay tests and state snapshots.
- [ ] Expand adversarial coverage to multi-step recovery and misleading
  completion claims, with native-language review.
- [ ] Ensure no benchmark examples are visible to model adapters before evaluation.

## P0 — Dataset and annotation (research blocker)

- [ ] Author 100–200 *distinct* realistic tasks, not merely 4 translations of the same examples.
- [ ] Sample Sindhi script, Roman Sindhi spellings, Urdu, and English; include mixed-language inputs.
- [ ] Document intent, canonical tool parameters, expected tool result, ambiguity policy, and safe response.
- [ ] Obtain the required rights and consent for source material; prohibit real account numbers or IDs.
- [ ] Use two independent people per verified task (annotator + separate reviewer). Resolve disagreements with an adjudication record, not silent overwrites.
- [ ] Report per-language reviewer agreement and spelling/dialect coverage.
- [ ] Split development and held-out test sets by *scenario group* to avoid translation leakage.
- [ ] Only release benchmark items labeled `verified` with review evidence; seed tasks remain `unverified`.

## P0 — Model support

- [x] Define a provider-agnostic conversation and tool schema with recorded tool traces.
- [x] OpenAI native adapter (offline SDK-contract tests; live validation pending).
- [x] Gemini native adapter (offline SDK-contract tests; live validation pending).
- [x] Claude native adapter (offline SDK-contract tests; live validation pending).
- [x] Hugging Face hosted Inference Providers native tool-call adapter (local Transformers pending).
- [x] Ollama native tool-call adapter (offline SDK-contract tests; local-model validation pending).
- [x] Normalize tool calls and record requested model, SDK version, decoding settings, task selection, dataset hash and elapsed time.\n- [ ] Add real-provider smoke tests, provider-returned model revision, token use, costs and retry policy.
- [ ] Repeat runs with fixed settings to measure variance. Never silently retry only unsuccessful models.

## P0 — Scoring protocol

- [x] Deterministic tool selection, parameter accuracy, and simulated execution
  outcomes, including expected rejection status and facts.
- [x] Preserve `null` for language fidelity and hallucination rate until independently reviewed.
- [ ] Validate the **natural-language final answer** against tool outputs through blinded native-language review.
- [ ] Measure unsupported claims, missing information, and inappropriate transaction confirmations.
- [x] Distinguish no-action safety from human-reviewed clarification/refusal correctness.
- [x] Enforce sandbox-level transfer confirmation policy with negative-action
  benchmark examples.
- [ ] Blinded native-language audits for confirmation and grounded explanations.
- [ ] Publish confidence intervals, raw denominators, human-review coverage, and failure taxonomy.

## P1 — Product and publication

- [ ] FastAPI service for benchmark results and tool schemas; do not expose real banking / government integrations.
- [ ] Streamlit leaderboard filtered by language, domain, model version, and evaluation date.
- [x] New run manifests with dataset hash, metadata and replayable JSONL logs; remaining: signed/immutable release process and independent scoring script.
- [ ] Research baseline report, annotation guidelines, benchmark card, known biases, and limitations.
- [ ] Publish reviewed benchmark dataset through Hugging Face Datasets and archive versioned releases.

## Current limitations (do not hide)

- The 16 initial examples are synthetic, **not human verified**. They exist to test the evaluator.
- The `oracle` command generates gold traces to smoke-test the harness, **not a real model score**.
- For successful actions, `task_completion` currently measures correct simulated
  execution plus nonempty final text; it does **not** verify answer truthfulness.
  For expected rejections and no-action tasks it requires human review.
- Each task starts a fresh **stateful** fictional sandbox; within-run calls
  deplete funds, inventory and seats. No real funds, orders or reservations exist.
- A correct mock rejection is measurable from tools, but task completion
  for a rejected operation remains unscored until human review.
- The 28 new scenarios are synthetic and not independently language-verified.
- The benchmark is not yet valid for claiming superior Sindhi-language performance by any model.

## Execution order (next)

1. [Issue #3](https://github.com/faheemkhaskheli9/Sindhi-AgentBench/issues/3): validate five providers against real tool-capable models, harden retry/usage handling, and evaluate local Transformers support.
2. [Issue #5](https://github.com/faheemkhaskheli9/Sindhi-AgentBench/issues/5): extend the
   implemented stateful sandbox with multi-step failure recovery and adversarial
   user confirmation workflows.
3. [Issue #4](https://github.com/faheemkhaskheli9/Sindhi-AgentBench/issues/4): blinded review and grounded final-answer evaluation.
4. [Issue #2](https://github.com/faheemkhaskheli9/Sindhi-AgentBench/issues/2): recruit native speakers, verify 100–200 original tasks and hold out scenario groups.
5. [Issues #6–7](https://github.com/faheemkhaskheli9/Sindhi-AgentBench/issues): leaderboard and research release after validation.
