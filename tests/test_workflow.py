from pathlib import Path

import pytest

from fakturama_automation.domain.errors import ManualReviewRequired, VerificationError
from decimal import Decimal

from fakturama_automation.domain.models import (
    DebtorCandidate,
    OrderInput,
    Payment,
    PaymentMethodCandidate,
    PaymentStatus,
    ProductCandidate,
    VatCandidate,
)
from fakturama_automation.gateways.simulated import SimulatedFakturamaGateway
from fakturama_automation.infrastructure.checkpoints import CheckpointStore
from fakturama_automation.infrastructure.evidence import EvidenceRecorder
from fakturama_automation.workflow.engine import WorkflowRunner, WorkflowState


def load_order() -> OrderInput:
    return OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )


def test_complete_missing_master_data_flow(tmp_path: Path) -> None:
    gateway = SimulatedFakturamaGateway()
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    result = runner.run(load_order())

    assert result.state is WorkflowState.FINAL_VERIFIED
    assert result.order_document.document_type == "Order"
    assert result.order_document.state == "open"
    assert result.invoice_document.document_type == "Invoice"
    assert result.invoice_document.state == "paid"
    assert result.order_document.transaction_id == result.invoice_document.transaction_id
    assert len(gateway.debtors) == 1
    assert len(gateway.payment_methods) == 1
    assert len(gateway.vats) == 2
    assert len(gateway.products) == 2
    assert gateway.events[:5] == [
        "open_new_order",
        "open_new_debtor",
        "create_payment_method",
        "select_debtor_payment_method",
        "save_debtor",
    ]
    assert gateway.events.index("save_order") < gateway.events.index(
        "create_follow_up_invoice"
    )
    assert gateway.events.index("create_follow_up_invoice") < gateway.events.index(
        "read_invoice_snapshot"
    )
    assert gateway.events.index("read_invoice_snapshot") < gateway.events.index(
        "set_invoice_payment"
    )
    assert gateway.events.index("set_invoice_payment") < gateway.events.index(
        "save_invoice"
    )
    for item in load_order().items:
        line_event = f"set_order_line:{item.sku}"
        assert gateway.events.index("create_vat") < gateway.events.index(line_event)
        assert gateway.events.index("create_product") < gateway.events.index(line_event)
    line_events = [
        index
        for index, event in enumerate(gateway.events)
        if event.startswith("set_order_line:")
    ]
    assert max(line_events) < gateway.events.index("complete_order")
    assert gateway.events.index("complete_order") < gateway.events.index("save_order")


def test_ambiguous_debtor_stops_for_manual_review(tmp_path: Path) -> None:
    order = load_order()
    candidate = DebtorCandidate(
        record_id="debtor-1",
        company=order.debtor.company,
        first_name=order.debtor.first_name,
        last_name=order.debtor.last_name,
        billing_address=order.debtor.billing_address,
    )
    duplicate = candidate.model_copy(update={"record_id": "debtor-2"})
    gateway = SimulatedFakturamaGateway(
        debtors=[(candidate, "Bank Transfer"), (duplicate, "Bank Transfer")]
    )
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    with pytest.raises(ManualReviewRequired):
        runner.run(order)

    assert runner.state is WorkflowState.MANUAL_REVIEW
    assert not gateway.documents


def test_existing_master_data_is_reused_and_unpaid_invoice_stays_unpaid(
    tmp_path: Path,
) -> None:
    order = load_order().model_copy(
        update={
            "payment": Payment(
                method="Bank Transfer", status=PaymentStatus.UNPAID
            )
        }
    )
    debtor = DebtorCandidate(
        record_id="debtor-existing",
        company=order.debtor.company,
        first_name=order.debtor.first_name,
        last_name=order.debtor.last_name,
        billing_address=order.debtor.billing_address,
    )
    payments = [
        PaymentMethodCandidate(
            record_id="payment-existing",
            name="Bank Transfer",
            description="Bank Transfer",
            payment_code="Credit transfer",
        )
    ]
    vats = [
        VatCandidate(
            record_id=f"vat-{value}",
            name=f"VAT {value}%",
            description=f"VAT {value}%",
            value=Decimal(value),
            e_invoice_code="S",
        )
        for value in ("19", "7")
    ]
    products = [
        ProductCandidate(
            record_id=f"product-{item.sku}",
            sku=item.sku,
            name=item.description,
            vat_percent=item.vat_percent,
            gross_price=Decimal("0"),
        )
        for item in order.items
    ]
    gateway = SimulatedFakturamaGateway(
        debtors=[(debtor, "Bank Transfer")],
        payment_methods=payments,
        vats=vats,
        products=products,
    )
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    result = runner.run(order)

    assert result.invoice_document.state == "unpaid"
    assert len(gateway.debtors) == 1
    assert len(gateway.payment_methods) == 1
    assert len(gateway.vats) == 2
    assert len(gateway.products) == 2


def test_available_exact_payment_option_is_used_without_opening_master_data(
    tmp_path: Path,
) -> None:
    conflict = PaymentMethodCandidate(
        record_id="payment-conflict",
        name="Bank Transfer",
        description="Bank Transfer",
        payment_code="Cash",
    )
    gateway = SimulatedFakturamaGateway(payment_methods=[conflict])
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    result = runner.run(load_order())

    assert result.state is WorkflowState.FINAL_VERIFIED
    assert "create_payment_method" not in gateway.events


def test_distinct_delivery_address_uses_main_invoice_address_and_continues(
    tmp_path: Path,
) -> None:
    order = load_order()
    delivery = order.debtor.billing_address.model_copy(
        update={"street": "Different Warehouse 44", "zip": "10553"}
    )
    order = order.model_copy(
        update={
            "debtor": order.debtor.model_copy(
                update={"delivery_address": delivery}
            )
        }
    )
    gateway = SimulatedFakturamaGateway()
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    result = runner.run(order)

    assert result.state is WorkflowState.FINAL_VERIFIED
    assert gateway.events.count("open_new_debtor") == 1
    saved_candidate, _ = next(iter(gateway.debtors.values()))
    assert saved_candidate.delivery_address is None
    assert {document.document_type for document in gateway.documents.values()} == {
        "Order",
        "Invoice",
    }


def test_explicit_identical_delivery_reuses_main_address_branch(tmp_path: Path) -> None:
    order = load_order()
    order = order.model_copy(
        update={
            "debtor": order.debtor.model_copy(
                update={"delivery_address": order.debtor.billing_address.model_copy()}
            )
        }
    )
    gateway = SimulatedFakturamaGateway()
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    result = runner.run(order)

    assert result.state is WorkflowState.FINAL_VERIFIED
    assert gateway.events.count("open_new_debtor") == 1


def test_same_debtor_identity_with_conflicting_address_requires_review(
    tmp_path: Path,
) -> None:
    order = load_order()
    conflicting_address = order.debtor.billing_address.model_copy(
        update={"zip": "99999", "city": "Elsewhere"}
    )
    candidate = DebtorCandidate(
        record_id="debtor-conflict",
        company=order.debtor.company,
        first_name=order.debtor.first_name,
        last_name=order.debtor.last_name,
        billing_address=conflicting_address,
    )
    gateway = SimulatedFakturamaGateway(
        debtors=[(candidate, order.payment.method)]
    )
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    with pytest.raises(ManualReviewRequired, match="conflicting ZIP or City"):
        runner.run(order)

    assert "open_new_debtor" not in gateway.events


def test_compatible_and_conflicting_same_name_vats_are_not_silently_reused(
    tmp_path: Path,
) -> None:
    order = load_order()
    label = f"VAT {order.items[0].vat_percent}%"
    vats = [
        VatCandidate(
            record_id="vat-good",
            name=label,
            description=label,
            value=order.items[0].vat_percent,
            e_invoice_code="S",
        ),
        VatCandidate(
            record_id="vat-conflict",
            name=label,
            description=label,
            value=Decimal("20"),
            e_invoice_code="S",
        ),
    ]
    gateway = SimulatedFakturamaGateway(vats=vats)
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    with pytest.raises(ManualReviewRequired, match="multiple or conflicting VAT"):
        runner.run(order)


def test_order_item_readback_must_match_source_before_save(tmp_path: Path) -> None:
    class CorruptingGateway(SimulatedFakturamaGateway):
        def read_order_snapshot(self):
            snapshot = super().read_order_snapshot()
            changed = snapshot.items[0].model_copy(update={"quantity": Decimal("999")})
            return snapshot.model_copy(update={"items": [changed, *snapshot.items[1:]]})

    gateway = CorruptingGateway()
    runner = WorkflowRunner(
        gateway,
        CheckpointStore(tmp_path / "checkpoint.json"),
        EvidenceRecorder(tmp_path),
    )

    with pytest.raises(VerificationError, match="Order item 0"):
        runner.run(load_order())

    assert "save_order" not in gateway.events
