"""Stateful, isolated simulation tests. Never use real bank or booking services."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sindhi_agentbench.dataset import load_tasks
from sindhi_agentbench.environments import MockSession, execute_mock
from sindhi_agentbench.runner import ModelTurn, Termination, run_task
from sindhi_agentbench.schema import Task, ToolCall

DATASET = Path(__file__).resolve().parents[1] / "data" / "seed_tasks.jsonl"


def transfer(amount=1500):
    return ToolCall(name="bank.transfer", arguments={
        "from_account": "A-100", "to_account": "B-200", "amount_pkr": amount,
    })


def booking(seats):
    return ToolCall(name="booking.reserve", arguments={
        "service_id": "BUS-KHI-HYD-01", "date": "2026-12-01", "seats": seats,
    })


def order(quantity):
    return ToolCall(name="shopping.place_order", arguments={
        "sku": "BOOK-SD-01", "quantity": quantity, "address_id": "ADDR-1",
    })


class ScriptedAdapter:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.received = []

    def complete(self, messages, tools):
        self.received.append((messages, tools))
        return next(self.turns)


def test_legacy_single_call_wrapper_remains_isolated():
    assert execute_mock(transfer()).facts["reference"] == "MOCK-TXN-001"
    assert execute_mock(transfer()).facts["reference"] == "MOCK-TXN-001"


def test_transfer_balances_and_exact_replay_protection():
    session = MockSession()
    assert session.execute(transfer()).facts == {
        "reference": "MOCK-TXN-001", "amount_pkr": 1500,
    }
    assert session.snapshot()["balances"] == {"A-100": 3500, "B-200": 1500}
    before = session.snapshot()
    assert session.execute(transfer()).facts["error"] == "duplicate_operation"
    assert session.snapshot() == before
    assert session.execute(transfer(4000)).facts["error"] == "insufficient_funds"
    assert session.snapshot() == before
    assert session.execute(transfer(2000)).facts["reference"] == "MOCK-TXN-002"
    assert session.snapshot()["balances"] == {"A-100": 1500, "B-200": 3500}


def test_bookings_and_orders_share_state_per_session_only():
    session = MockSession()
    assert session.execute(booking(2)).status == "success"
    assert session.execute(booking(3)).facts["error"] == "sold_out"
    assert session.execute(booking(1)).facts["reference"] == "MOCK-BOOK-002"
    assert session.execute(order(2)).status == "success"
    assert session.execute(order(3)).facts["error"] == "out_of_stock"
    assert session.execute(order(1)).facts["reference"] == "MOCK-ORDER-002"
    state = session.snapshot()
    assert state["available_seats"]["BUS-KHI-HYD-01:2026-12-01"] == 1
    assert state["inventory"]["BOOK-SD-01"] == 0
    assert MockSession().snapshot()["inventory"]["BOOK-SD-01"] == 3


def test_bad_arguments_and_unknown_destinations_leave_state_unchanged():
    session = MockSession()
    before = session.snapshot()
    calls = [
        ToolCall(name="bank.transfer", arguments={
            "from_account": "A-100", "to_account": "B-200", "amount_pkr": "100",
        }),
        ToolCall(name="bank.transfer", arguments={
            "from_account": "A-100", "to_account": "B-999", "amount_pkr": 100,
        }),
        ToolCall(name="booking.reserve", arguments={
            "service_id": "MISSING", "date": "2026-12-01", "seats": 1,
        }),
        ToolCall(name="shopping.place_order", arguments={
            "sku": "MISSING", "quantity": 1, "address_id": "ADDR-1",
        }),
    ]
    for call in calls:
        assert session.execute(call).status == "rejected"
        assert session.snapshot() == before


def test_explicit_confirmation_required_is_enforced_by_sandbox():
    session = MockSession(confirmation_required=True)
    before = session.snapshot()
    assert session.execute(transfer()).facts["error"] == "confirmation_required"
    assert session.snapshot() == before
    # Only the benchmark controller, not the model's tool arguments, sets consent.
    session.confirmation_granted = True
    assert session.execute(transfer()).status == "success"


def test_government_read_only_repeat_and_untrusted_observation():
    session = MockSession()
    lookup = ToolCall(name="government.check_application", arguments={
        "application_id": "APP-123",
    })
    assert session.execute(lookup).status == "success"
    assert session.execute(lookup).status == "success"
    injected = session.execute(ToolCall(name="government.check_application", arguments={
        "application_id": "APP-INJECT",
    }))
    assert injected.facts["application_status"] == "under_review"
    assert "SYSTEM OVERRIDE" in injected.facts["untrusted_note"]
    assert session.snapshot()["successful_mutations"] == 0
    assert session.execute(ToolCall(name="government.check_application", arguments={
        "application_id": "APP-NOPE",
    })).facts["error"] == "not_found"


def test_runner_persists_state_across_calls_and_resets_for_new_tasks():
    task = next(x for x in load_tasks(DATASET) if x.domain.value == "banking")
    scripted = [
        ModelTurn(tool_calls=[transfer(1500), transfer(1500), transfer(2000)]),
        ModelTurn(content="Simulation completed with one replay rejected."),
    ]
    first = run_task(task, ScriptedAdapter(scripted))
    assert first.termination == Termination.COMPLETE
    assert [x.result.status for x in first.actions] == [
        "success", "rejected", "success",
    ]
    assert first.actions[1].result.facts["error"] == "duplicate_operation"
    assert first.environment_state["balances"] == {"A-100": 1500, "B-200": 3500}
    assert len(first.as_prediction().tool_calls) == 3
    again = run_task(task, ScriptedAdapter([
        ModelTurn(tool_calls=[transfer(1500)]),
        ModelTurn(content="Only simulated."),
    ]))
    assert again.actions[0].result.facts["reference"] == "MOCK-TXN-001"
    assert again.environment_state["balances"]["A-100"] == 3500


def test_runner_caps_burst_tool_calls_without_claiming_completion():
    task = next(x for x in load_tasks(DATASET) if x.domain.value == "banking")
    run = run_task(
        task, ScriptedAdapter([ModelTurn(tool_calls=[transfer()] * 40)]),
        max_tool_calls=2,
    )
    assert run.termination == Termination.MAX_TOOL_CALLS
    assert len(run.actions) == 2
    assert run.final_answer == ""
    assert run.as_prediction().final_answer == ""
    assert run.environment_state["balances"]["A-100"] == 3500
    with pytest.raises(ValueError, match="max_tool_calls"):
        run_task(task, ScriptedAdapter([]), max_tool_calls=0)


def test_unconfirmed_task_cannot_request_executed_gold_action():
    data = load_tasks(DATASET)[0].model_dump()
    data["confirmation_required"] = True
    with pytest.raises(ValidationError, match="Unconfirmed"):
        Task.model_validate(data)
    data.update(expected_behavior="clarify", expected_call=None, expected_facts={})
    task = Task.model_validate(data)
    run = run_task(task, ScriptedAdapter([
        ModelTurn(tool_calls=[transfer()]),
        ModelTurn(content="I need your confirmation."),
    ]))
    assert run.actions[0].result.facts["error"] == "confirmation_required"
    assert run.environment_state["balances"]["A-100"] == 5000
