from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .calculations import calculated_totals, line_net, within_tolerance
from .models import OrderInput, PaymentStatus


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    severity: Severity = Severity.ERROR
    code: str = "VALIDATION_ISSUE"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def requires_review(self) -> bool:
        """Warnings are safe to extract, but not safe to write unattended."""
        return bool(self.issues)

    def raise_for_errors(self) -> None:
        if not self.valid:
            details = "; ".join(f"{item.path}: {item.message}" for item in self.issues)
            raise OrderValidationError(details)


class OrderValidationError(ValueError):
    pass


def validate_order(order: OrderInput, *, require_evidence: bool = False) -> ValidationReport:
    report = ValidationReport()

    if order.payment.status is PaymentStatus.PAID and order.payment.payment_date is None:
        report.issues.append(
            ValidationIssue(
                code="PAYMENT_DATE_REQUIRED",
                path="payment.payment_date",
                message="a PAID source state requires a printed payment date",
            )
        )
    if order.payment.status is PaymentStatus.UNPAID and order.payment.payment_date is not None:
        report.issues.append(
            ValidationIssue(
                code="PAYMENT_DATE_CONFLICT",
                path="payment.payment_date",
                message="an UNPAID source state contains a payment date and requires review",
            )
        )
    if order.payment.status not in {PaymentStatus.PAID, PaymentStatus.UNPAID}:
        report.issues.append(
            ValidationIssue(
                code="UNSUPPORTED_PAYMENT_STATUS",
                path="payment.status",
                message=(
                    f"source payment state {order.payment.status.value!r} is preserved but "
                    "has no authorized Fakturama workflow mapping"
                ),
            )
        )

    for index, item in enumerate(order.items):
        expected_line_total = line_net(item)
        if not within_tolerance(expected_line_total, item.source_total):
            report.issues.append(
                ValidationIssue(
                    code="LINE_TOTAL_MISMATCH",
                    path=f"items[{index}].source_total",
                    message=(
                        f"source line total {item.source_total} does not match "
                        f"calculated net line total {expected_line_total}"
                    ),
                )
            )

    expected = calculated_totals(order)
    for field_name in ("total_net", "total_vat", "total_gross"):
        expected_value = getattr(expected, field_name)
        actual_value = getattr(order.totals, field_name)
        if not within_tolerance(expected_value, actual_value):
            report.issues.append(
                ValidationIssue(
                    code="ORDER_TOTAL_MISMATCH",
                    path=f"totals.{field_name}",
                    message=f"source {actual_value} does not match calculated {expected_value}",
                )
            )

    for path, evidence in order.evidence.items():
        if evidence.confidence < 0.75:
            report.issues.append(
                ValidationIssue(
                    code="LOW_OCR_CONFIDENCE",
                    path=path,
                    message=f"low extraction confidence ({evidence.confidence:.2f})",
                )
            )
        elif evidence.confidence < 0.90:
            report.issues.append(
                ValidationIssue(
                    code="MODERATE_OCR_CONFIDENCE",
                    path=path,
                    message=f"moderate extraction confidence ({evidence.confidence:.2f})",
                    severity=Severity.WARNING,
                )
            )

    if require_evidence:
        missing = sorted(_critical_evidence_paths(order) - order.evidence.keys())
        if missing:
            report.issues.append(
                ValidationIssue(
                    code="MISSING_CRITICAL_EVIDENCE",
                    path="evidence",
                    message="missing grounded evidence for: " + ", ".join(missing),
                )
            )

    return report


def _critical_evidence_paths(order: OrderInput) -> set[str]:
    paths = {
        "order_date",
        "external_reference",
        "payment.method",
        "payment.status",
        "totals.total_net",
        "totals.total_vat",
        "totals.total_gross",
        "debtor.billing_address.zip",
        "debtor.billing_address.city",
    }
    if order.debtor.company:
        paths.add("debtor.company")
    if order.debtor.first_name:
        paths.add("debtor.first_name")
    if order.debtor.last_name:
        paths.add("debtor.last_name")
    if order.payment.payment_date:
        paths.add("payment.payment_date")
    for index, _ in enumerate(order.items):
        paths.update(
            {
                f"items[{index}].sku",
                f"items[{index}].quantity",
                f"items[{index}].unit_net_price",
                f"items[{index}].vat_percent",
                f"items[{index}].discount_percent",
                f"items[{index}].source_total",
            }
        )
    return paths
