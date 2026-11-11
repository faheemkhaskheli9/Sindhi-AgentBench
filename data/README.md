# Data policy

`seed_tasks.jsonl` contains synthetic fixtures for testing the harness, **not published benchmark gold data**. The language strings and canonical actions have not been reviewed by native-language annotators. Do not train on these examples or use oracle scores in a leaderboard.

Planned reviewed dataset: 100–200 scenarios with source provenance, independent annotation and adjudication, dialect and spelling variation, and scenario-grouped held-out splits. Personal, payment, booking, and government IDs must always be fictional. See `docs/ROADMAP.md`.

Additional unverified fixture datasets:

- `edge_cases.jsonl`: 8 clarification/refusal requests (no tool call allowed).
- `confirmation_tasks.jsonl`: 4 transfers requiring confirmation first.
- `scenario_tasks.jsonl`: 24 expected success/failure and prompt-injection
  cases across the four supported languages.

See `SCENARIOS.md` and `EDGE_CASES.md` for expected policy and
human-review limitations. Fixture count: 52; human verified: 0.
