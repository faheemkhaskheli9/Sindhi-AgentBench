from sindhi_agentbench.environments import execute_mock
from sindhi_agentbench.schema import ToolCall


def test_valid_mock_bank_transfer():
    call = ToolCall(name="bank.transfer", arguments={
        "from_account": "A-100", "to_account": "B-200", "amount_pkr": 1500
    })
    assert execute_mock(call).facts == {"reference": "MOCK-TXN-001", "amount_pkr": 1500}


def test_strict_argument_types_and_extra_keys():
    data = {"from_account": "A-100", "to_account": "B-200", "amount_pkr": "1500"}
    assert execute_mock(ToolCall(name="bank.transfer", arguments=data)).status == "rejected"
    data["amount_pkr"] = 1500
    data["extra"] = "unexpected"
    assert execute_mock(ToolCall(name="bank.transfer", arguments=data)).status == "rejected"


def test_invalid_tools_do_not_execute():
    assert execute_mock(ToolCall(name="bank.withdraw_real_funds", arguments={})).facts == {
        "error": "unknown_tool"
    }


def test_guardrails_for_all_mock_domains():
    assert execute_mock(ToolCall(name="bank.transfer", arguments={
        "from_account": "A-100", "to_account": "B-200", "amount_pkr": 9000
    })).facts["error"] == "insufficient_funds"
    assert execute_mock(ToolCall(name="booking.reserve", arguments={
        "service_id": "BUS-KHI-HYD-01", "date": "2026-12-01", "seats": 5
    })).facts["error"] == "sold_out"
    assert execute_mock(ToolCall(name="shopping.place_order", arguments={
        "sku": "BOOK-SD-01", "quantity": 4, "address_id": "ADDR-1"
    })).facts["error"] == "out_of_stock"
    assert execute_mock(ToolCall(name="government.check_application", arguments={
        "application_id": "UNKNOWN"
    })).facts["error"] == "not_found"
