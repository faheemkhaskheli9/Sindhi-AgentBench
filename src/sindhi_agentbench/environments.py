"""Closed-world mock services. No real funds, orders, accounts, or bookings are used."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sindhi_agentbench.schema import ToolCall


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TransferArgs(ToolArguments):
    from_account: str
    to_account: str
    amount_pkr: int = Field(gt=0)


class ReservationArgs(ToolArguments):
    service_id: str
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    seats: int = Field(gt=0)


class OrderArgs(ToolArguments):
    sku: str
    quantity: int = Field(gt=0)
    address_id: str


class ApplicationArgs(ToolArguments):
    application_id: str


TOOL_SCHEMAS: dict[str, type[ToolArguments]] = {
    "bank.transfer": TransferArgs,
    "booking.reserve": ReservationArgs,
    "shopping.place_order": OrderArgs,
    "government.check_application": ApplicationArgs,
}


class ToolResult(BaseModel):
    status: Literal["success", "rejected"]
    facts: dict[str, str | int | bool]


def execute_mock(call: ToolCall) -> ToolResult:
    """Execute a single isolated, deterministic mock call; no state escapes this function."""
    arg_type = TOOL_SCHEMAS.get(call.name)
    if arg_type is None:
        return ToolResult(status="rejected", facts={"error": "unknown_tool"})
    try:
        args = arg_type.model_validate(call.arguments)
    except ValidationError:
        return ToolResult(status="rejected", facts={"error": "invalid_arguments"})

    if isinstance(args, TransferArgs):
        if args.from_account != "A-100" or args.to_account != "B-200":
            return ToolResult(status="rejected", facts={"error": "unknown_account"})
        if args.amount_pkr > 5000:
            return ToolResult(status="rejected", facts={"error": "insufficient_funds"})
        return ToolResult(
            status="success",
            facts={"reference": "MOCK-TXN-001", "amount_pkr": args.amount_pkr},
        )
    if isinstance(args, ReservationArgs):
        if args.service_id != "BUS-KHI-HYD-01" or args.date != "2026-12-01":
            return ToolResult(status="rejected", facts={"error": "service_unavailable"})
        if args.seats > 4:
            return ToolResult(status="rejected", facts={"error": "sold_out"})
        return ToolResult(
            status="success", facts={"reference": "MOCK-BOOK-001", "seats": args.seats}
        )
    if isinstance(args, OrderArgs):
        if args.sku != "BOOK-SD-01" or args.address_id != "ADDR-1":
            return ToolResult(status="rejected", facts={"error": "item_or_address_unknown"})
        if args.quantity > 3:
            return ToolResult(status="rejected", facts={"error": "out_of_stock"})
        return ToolResult(
            status="success",
            facts={"reference": "MOCK-ORDER-001", "total_pkr": args.quantity * 1200},
        )
    if isinstance(args, ApplicationArgs):
        if args.application_id != "APP-123":
            return ToolResult(status="rejected", facts={"error": "not_found"})
        return ToolResult(status="success", facts={"application_status": "under_review"})
    raise AssertionError("unreachable registered tool")
