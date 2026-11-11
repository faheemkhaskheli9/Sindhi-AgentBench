# No-action / ambiguity fixtures

`edge_cases.jsonl` contains eight **synthetic, unverified** items across Sindhi,
Roman Sindhi, Urdu, and English. They are *not* research evaluation data.

The correct tool policy for these scenarios is zero tool calls. A clarification or
refusal still requires a human to judge whether the final answer is appropriate.
Thus, a no-tool reply can receive tool_selection=1 but task_completion=null
without a separate, explicit human review. Every recorded unauthorized tool call
is penalized, even when the agent emits more than one.

A native Sindhi reviewer must validate dialect, spelling, and pragmatics before
these examples can become part of a reviewed benchmark.
