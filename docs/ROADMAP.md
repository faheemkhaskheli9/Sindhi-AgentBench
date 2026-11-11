# Development plan and release gates

## Goal

Evaluate **real task execution**, not generic fluency or multiple-choice questions. A model must interpret a Sindhi, Roman Sindhi, Urdu, or English request; select the correct tool; supply correct arguments; execute against isolated mock services; and accurately explain the verified outcome.

## P0 — Reproducible harness (initial implementation)

- [x] Strict versioned JSONL contracts for tasks, prediction traces, and separate human reviews.
- [x] Isolated deterministic tool simulators for banking, booking, shopping, and government services.
- [x] Offline tool-selection, exact-argument, and mock-execution scoring, stratified by language.
- [x] Synthetic starter tasks, unit tests, CLI, and GitHub Actions CI.
- [ ] Integrate model-generated tool calls and post-tool explanations (multi-turn tool-use loop).
- [ ] Add refusal, ambiguous-intent, code-switching, prompt-injection, and tool-failure scenarios.
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

- [ ] Define a provider-agnostic conversation and tool schema with audit logs.
- [ ] OpenAI adapter.
- [ ] Gemini adapter.
- [ ] Claude adapter.
- [ ] Hugging Face Transformers adapter.
- [ ] Ollama adapter.
- [ ] Normalize equivalent tool-call events; record model, version, provider, decoding settings, retries, latency, and price at evaluation time.
- [ ] Repeat runs with fixed settings to measure variance. Never silently retry only unsuccessful models.

## P0 — Scoring protocol

- [x] Deterministic tool selection, parameter accuracy, and simulated execution success.
- [x] Preserve `null` for language fidelity and hallucination rate until independently reviewed.
- [ ] Validate the **natural-language final answer** against tool outputs through blinded native-language review.
- [ ] Measure unsupported claims, missing information, and inappropriate transaction confirmations.
- [ ] Distinguish task completion from refusal correctness and safety-critical confirmation behavior.
- [ ] Publish confidence intervals, raw denominators, human-review coverage, and failure taxonomy.

## P1 — Product and publication

- [ ] FastAPI service for benchmark results and tool schemas; do not expose real banking / government integrations.
- [ ] Streamlit leaderboard filtered by language, domain, model version, and evaluation date.
- [ ] Immutable run manifests and replayable JSONL logs; release an independent scoring script.
- [ ] Research baseline report, annotation guidelines, benchmark card, known biases, and limitations.
- [ ] Publish reviewed benchmark dataset through Hugging Face Datasets and archive versioned releases.

## Current limitations (do not hide)

- The 16 initial examples are synthetic, **not human verified**. They exist to test the evaluator.
- The `oracle` command generates gold traces to smoke-test the harness, **not a real model score**.
- `task_completion` currently means correct simulated action plus a non-empty final answer; it does **not** verify answer truthfulness or language, which are reviewed separately.
- Every mock execution is independent and stateless, so repeat calls do not deplete inventory or transfer real value.
- The benchmark is not yet valid for claiming superior Sindhi-language performance by any model.
