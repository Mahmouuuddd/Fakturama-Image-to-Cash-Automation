from pathlib import Path
from decimal import Decimal

from fakturama_automation.domain.models import OrderInput, PaymentStatus
from fakturama_automation.domain.validation import validate_order


def test_example_order_reconciles() -> None:
    order = OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )
    assert validate_order(order).valid


def test_total_mismatch_is_rejected() -> None:
    order = OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )
    broken = order.model_copy(
        update={
            "totals": order.totals.model_copy(
                update={"total_gross": Decimal("999.99")}
            )
        }
    )
    report = validate_order(broken)

    assert not report.valid
    assert any(issue.path == "totals.total_gross" for issue in report.issues)


def test_paid_without_date_is_a_policy_issue_not_a_model_crash() -> None:
    order = OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )
    order = order.model_copy(
        update={"payment": order.payment.model_copy(update={"payment_date": None})}
    )

    report = validate_order(order)

    assert any(issue.code == "PAYMENT_DATE_REQUIRED" for issue in report.issues)


def test_source_payment_state_is_preserved_for_manual_policy_mapping() -> None:
    order = OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )
    order = order.model_copy(
        update={
            "payment": order.payment.model_copy(
                update={"status": PaymentStatus.PARTIALLY_PAID}
            )
        }
    )

    report = validate_order(order)

    assert order.payment.status is PaymentStatus.PARTIALLY_PAID
    assert any(issue.code == "UNSUPPORTED_PAYMENT_STATUS" for issue in report.issues)


def test_unusual_but_valid_numeric_values_reconcile_locally() -> None:
    order = OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )
    item = order.items[0].model_copy(
        update={
            "quantity": Decimal("0.75"),
            "unit_net_price": Decimal("1000"),
            "discount_percent": Decimal("100"),
            "vat_percent": Decimal("7.7"),
            "source_total": Decimal("0"),
        }
    )
    order = order.model_copy(
        update={
            "items": [item],
            "totals": order.totals.model_copy(
                update={
                    "total_net": Decimal("0"),
                    "total_vat": Decimal("0"),
                    "total_gross": Decimal("0"),
                }
            ),
        }
    )

    assert validate_order(order).valid
