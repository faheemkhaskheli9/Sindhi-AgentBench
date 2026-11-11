# Synthetic adverse outcomes

`scenario_tasks.jsonl` contains 24 **unverified** cases (4 languages × 6
scenarios): insufficient funds, sold-out bookings, out-of-stock shopping,
missing applications, invalid accounts, and an untrusted government-service
note attempting prompt injection. Some Roman Sindhi cases use code switching.

`confirmation_tasks.jsonl` contains 4 **unverified** tasks that require
the agent to ask for explicit consent **without calling a transfer tool**.
The sandbox itself rejects an unconfirmed transfer.

All identifiers are fictional. Every model run starts with fresh bank balances,
booking capacity, inventory and application state. Mutations persist **within**
one run, and repeated successful mutations are rejected deterministically.

For expected service rejections, automatic `execution_outcome` is computable,
but `task_completion` remains `null` until an independent human checks the
agent's final explanation. Injection cases also need human hallucination review:
a correct call alone does **not** verify answer faithfulness.

Neither dataset is eligible for any research leaderboard until native-speaking
human annotation and independent adjudication are complete.
