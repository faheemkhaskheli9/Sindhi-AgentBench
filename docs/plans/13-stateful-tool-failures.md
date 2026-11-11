# Phase 3 — Stateful and adversarial tool-use evaluation

Status: **Implemented foundation; human review and richer scenarios remain P0**.
Owner: Sindhi-AgentBench. Existing issue: [#5](../../issues/5).

## Why

The original simulator recreated balances, seat capacity and inventory on
every call. An agent could repeat a transfer without consuming its balance,
and failure cases could not be represented as gold tasks. That was misleading
for end-to-end task reliability.

## Delivered

1. **Per-run state:** each agent run constructs its own in-memory `MockSession`.
   Successful bank transfers, reservations and orders deplete fictional
   balances, seats and stock; another run starts from the same initial state.
2. **Replay protection:** an identical successful mutation is rejected as
   `duplicate_operation`. Read-only government lookups can be repeated.
   Failed mutations have no side effects or consumed replay keys.
3. **Consent:** a task may require explicit transfer confirmation. The simulator
   rejects unauthorized transfers without mutating balances. No model has a
   tool allowing it to grant itself consent.
4. **Bounded execution:** agent runs have a configurable `max_tool_calls`
   total action limit (default 12), independent of the conversation-turn limit.
5. **Deterministic failures:** dataset gold accepts expected service rejections.
   `execution_outcome` checks correct tool, exact arguments, status and facts.
   Correct rejection **does not** imply end-to-end completion; a human must
   confirm that the model described the failure honestly.
6. **Untrusted observations:** APP-INJECT returns a fictional application state
   with a malicious-looking note in tool data. The actual state is still
   `under_review`. Semantic prompt-injection resistance needs human scoring.
7. **Multilingual fixtures:** 24 synthetic expected-outcome scenarios and
   4 confirmation-required requests, in addition to the 24 existing seed and
   no-action tasks. None are yet language-verified.
8. **Replay evidence:** save mutable state snapshots with run traces and
   retain original task/prompt isolation.

## Test and acceptance checks

- Existing seed/edge fixtures stay valid.
- Transfers, bookings and orders deplete state; new sessions are independent.
- Failed calls and duplicates leave state unchanged.
- Unauthorized transfers never spend funds.
- More than the allowed tool-call count ends the run without completion credit.
- Missing/invalid arguments and false gold expectations are rejected.
- English/Urdu/Sindhi/Roman Sindhi synthetic fixtures validate under the same
  Pydantic schemas and service contracts.
- Every run stays labelled `leaderboard_eligible=false` until reviewed.

## Next plan (do not duplicate existing backlog)

Continue [#5](https://github.com/faheemkhaskheli9/Sindhi-AgentBench/issues/5)
with adversarial multi-step failures (retries after partial success, realistic
customer confirmations, and tool-output injection controls). Continue
[#4](https://github.com/faheemkhaskheli9/Sindhi-AgentBench/issues/4) with
blinded review of grounded final answers. Block publication under
[#2](https://github.com/faheemkhaskheli9/Sindhi-AgentBench/issues/2) until
independent Sindhi-language annotation and scenario-grouped splits exist.

This is a mock research environment, not a payment or government system.
