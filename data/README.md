# Data policy

`seed_tasks.jsonl` contains synthetic fixtures for testing the harness, **not published benchmark gold data**. The language strings and canonical actions have not been reviewed by native-language annotators. Do not train on these examples or use oracle scores in a leaderboard.

Planned reviewed dataset: 100–200 scenarios with source provenance, independent annotation and adjudication, dialect and spelling variation, and scenario-grouped held-out splits. Personal, payment, booking, and government IDs must always be fictional. See `docs/ROADMAP.md`.
