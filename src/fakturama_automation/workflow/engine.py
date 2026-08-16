from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from fakturama_automation.domain.calculations import product_gross_price, within_tolerance
from fakturama_automation.domain.errors import ManualReviewRequired, VerificationError
from fakturama_automation.domain.matching import (
    addresses_match,
    debtor_matches,
    exact_matches,
    normalize_text,
    product_matches,
)
from fakturama_automation.domain.models import (
    DocumentRecord,
    OrderInput,
    PaymentMethodCandidate,
    ProductCandidate,
    VatCandidate,
)
from fakturama_automation.domain.validation import validate_order
from fakturama_automation.gateways.base import FakturamaGateway
from fakturama_automation.infrastructure.checkpoints import CheckpointStore
from fakturama_automation.infrastructure.evidence import EvidenceRecorder


PAYMENT_CODES = {
    "bank transfer": "Credit transfer",
    "credit card": "Credit card",
    "sepa direct debit": "SEPA direct debit",
}


class WorkflowState(StrEnum):
    CREATED = "CREATED"
    OCR_COMPLETED = "OCR_COMPLETED"
    EXTRACTED = "EXTRACTED"
    VALIDATED = "VALIDATED"
    ORDER_OPEN = "ORDER_OPEN"
    DEBTOR_RESOLVED = "DEBTOR_RESOLVED"
    ITEMS_RESOLVED = "ITEMS_RESOLVED"
    ORDER_VALIDATED = "ORDER_VALIDATED"
    ORDER_SAVED = "ORDER_SAVED"
    ORDER_VERIFIED = "ORDER_VERIFIED"
    INVOICE_OPEN = "INVOICE_OPEN"
    PAYMENT_APPLIED = "PAYMENT_APPLIED"
    INVOICE_SAVED = "INVOICE_SAVED"
    FINAL_VERIFIED = "FINAL_VERIFIED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    FAILED = "FAILED"


@dataclass(frozen=True)
class WorkflowResult:
    state: WorkflowState
    order_document: DocumentRecord
    invoice_document: DocumentRecord
    run_directory: Path


class WorkflowRunner:
    def __init__(
        self,
        gateway: FakturamaGateway,
        checkpoint_store: CheckpointStore,
        evidence: EvidenceRecorder,
        *,
        workflow_id: str | None = None,
        initial_state: WorkflowState = WorkflowState.CREATED,
    ) -> None:
        self.gateway = gateway
        self.checkpoint_store = checkpoint_store
        self.evidence = evidence
        self.workflow_id = workflow_id
        self.state = initial_state
        self.external_reference: str | None = None
        self.order_number: str | None = None
        self.invoice_number: str | None = None
        self.order_document: DocumentRecord | None = None
        self.invoice_document: DocumentRecord | None = None

    def run(self, order: OrderInput) -> WorkflowResult:
        self.external_reference = order.external_reference
        try:
            report = validate_order(order)
            report.raise_for_errors()
            self._transition(WorkflowState.VALIDATED, warnings=len(report.issues))

            self.gateway.preflight()
            self.order_number = self.gateway.open_new_order()
            self.gateway.set_order_header(order)
            self._transition(WorkflowState.ORDER_OPEN, order_number=self.order_number)

            self._resolve_debtor(order)
            self._transition(WorkflowState.DEBTOR_RESOLVED)

            for index, item in enumerate(order.items):
                self._resolve_product(item, index)
            self._transition(WorkflowState.ITEMS_RESOLVED, item_count=len(order.items))

            # Take-home specification §4.2: apply document-level defaults only
            # after every Product line has been completed.
            self.gateway.complete_order()
            self._verify_order_editor(order)
            self._transition(WorkflowState.ORDER_VALIDATED)

            self.order_document = self.gateway.save_order()
            self._transition(
                WorkflowState.ORDER_SAVED,
                document=self.order_document.model_dump(mode="json"),
            )
            persisted_order = self.gateway.verify_document(self.order_document)
            self._verify_document_identity(self.order_document, persisted_order)
            self._transition(WorkflowState.ORDER_VERIFIED)

            self.invoice_number = self.gateway.create_follow_up_invoice()
            self._verify_linked_invoice(order)
            self._transition(WorkflowState.INVOICE_OPEN, invoice_number=self.invoice_number)

            self.gateway.set_invoice_payment(order.payment, order.totals.total_gross)
            self._transition(WorkflowState.PAYMENT_APPLIED)

            self.invoice_document = self.gateway.save_invoice()
            self._transition(
                WorkflowState.INVOICE_SAVED,
                document=self.invoice_document.model_dump(mode="json"),
            )
            persisted_invoice = self.gateway.verify_document(self.invoice_document)
            self._verify_document_identity(self.invoice_document, persisted_invoice)

            # The source Order must still be present and open after Invoice creation.
            persisted_order = self.gateway.verify_document(self.order_document)
            self._verify_document_identity(self.order_document, persisted_order)
            if normalize_text(persisted_order.state) != "open":
                raise VerificationError(
                    f"source Order state is {persisted_order.state!r}, expected 'open'"
                )
            if (
                persisted_invoice.transaction_id
                != persisted_order.transaction_id
            ):
                raise VerificationError(
                    "verified Invoice is not linked to the source Order transaction"
                )
            self._transition(WorkflowState.FINAL_VERIFIED)

            return WorkflowResult(
                state=self.state,
                order_document=self.order_document,
                invoice_document=self.invoice_document,
                run_directory=self.evidence.run_directory,
            )
        except ManualReviewRequired as exc:
            prior_state = self.state
            self._transition(
                WorkflowState.MANUAL_REVIEW,
                error=str(exc),
                prior_state=prior_state.value,
            )
            raise
        except Exception as exc:
            self._transition(WorkflowState.FAILED, error=str(exc), error_type=type(exc).__name__)
            raise

    def _resolve_debtor(self, order: OrderInput) -> None:
        debtor = order.debtor
        query = debtor.company or " ".join((debtor.first_name, debtor.last_name)).strip()
        candidates = self.gateway.search_debtors(query)
        matches = exact_matches(candidates, lambda candidate: debtor_matches(debtor, candidate))
        identity_matches = [
            candidate
            for candidate in candidates
            if normalize_text(candidate.company) == normalize_text(debtor.company)
            and normalize_text(candidate.first_name) == normalize_text(debtor.first_name)
            and normalize_text(candidate.last_name) == normalize_text(debtor.last_name)
        ]
        conflicting_identity = [
            candidate for candidate in identity_matches if candidate not in matches
        ]
        if conflicting_identity:
            self.gateway.cancel_active_dialog()
            raise ManualReviewRequired(
                f"Debtor search for {query!r} contains the same identity with "
                "conflicting ZIP or City: "
                + ", ".join(candidate.record_id for candidate in conflicting_identity)
            )
        if len(matches) > 1:
            self.gateway.cancel_active_dialog()
            raise ManualReviewRequired(
                f"multiple Debtors exactly match {query!r}: "
                + ", ".join(match.record_id for match in matches)
            )

        if not matches:
            self.gateway.cancel_active_dialog()
            # Take-home specification §2.5–§2.11: open and fill the Debtor
            # first, then resolve its Payment Method while both Debtor and
            # source Order editors remain open.
            self.gateway.open_new_debtor(debtor)
            if not self.gateway.select_debtor_payment_method(order.payment.method):
                self._ensure_payment_method(order.payment.method)
                if not self.gateway.select_debtor_payment_method(order.payment.method):
                    raise ManualReviewRequired(
                        f"Payment Method {order.payment.method!r} could not be selected "
                        "in the open Debtor after exact creation/reuse"
                    )
            self.gateway.save_debtor()
            candidates = self.gateway.search_debtors(query)
            matches = exact_matches(
                candidates, lambda candidate: debtor_matches(debtor, candidate)
            )
            if len(matches) != 1:
                raise ManualReviewRequired(
                    f"newly created Debtor {query!r} could not be reselected unambiguously"
                )

        self.gateway.select_debtor(matches[0].record_id)
        selected = self.gateway.read_selected_debtor()
        if not addresses_match(debtor.billing_address, selected.billing_address):
            raise VerificationError("selected Debtor billing address differs from source")
        if not addresses_match(
            debtor.effective_delivery_address, selected.effective_delivery_address
        ):
            raise VerificationError("selected Debtor delivery address differs from source")

    def _ensure_payment_method(self, method: str) -> None:
        candidates = self.gateway.search_payment_methods(method)
        same_name = [
            candidate
            for candidate in candidates
            if normalize_text(candidate.name) == normalize_text(method)
        ]
        if len(same_name) > 1:
            raise ManualReviewRequired(f"multiple payment methods match {method!r}")

        expected_code = PAYMENT_CODES.get(normalize_text(method))
        if same_name:
            candidate = same_name[0]
            if normalize_text(candidate.description) != normalize_text(method):
                raise ManualReviewRequired(
                    f"payment method {method!r} has conflicting description "
                    f"{candidate.description!r}"
                )
            if expected_code and normalize_text(candidate.payment_code) != normalize_text(
                expected_code
            ):
                raise ManualReviewRequired(
                    f"payment method {method!r} has conflicting code "
                    f"{candidate.payment_code!r}"
                )
            return

        if expected_code is None:
            raise ManualReviewRequired(
                f"payment method {method!r} is missing and has no approved creation mapping"
            )

        self.gateway.create_payment_method(
            PaymentMethodCandidate(
                record_id="",
                name=method,
                description=method,
                payment_code=expected_code,
                cash_discount=Decimal("0"),
                discount_days=0,
                net_days=0,
            )
        )

    def _resolve_product(self, item, source_index: int) -> None:
        candidates = self.gateway.search_products(item.sku)
        matches = exact_matches(candidates, lambda candidate: product_matches(item.sku, candidate))
        if len(matches) > 1:
            self.gateway.cancel_active_dialog()
            raise ManualReviewRequired(f"multiple Products have exact SKU {item.sku!r}")

        if not matches:
            self.gateway.cancel_active_dialog()
            vat_record_id = self._ensure_vat(item.vat_percent)
            self.gateway.create_product(
                ProductCandidate(
                    record_id="",
                    sku=item.sku,
                    name=item.description,
                    vat_percent=item.vat_percent,
                    gross_price=product_gross_price(item),
                )
            )
            candidates = self.gateway.search_products(item.sku)
            matches = exact_matches(
                candidates, lambda candidate: product_matches(item.sku, candidate)
            )
            if len(matches) != 1:
                raise ManualReviewRequired(
                    f"new Product {item.sku!r} could not be reselected unambiguously; "
                    f"VAT record was {vat_record_id}"
                )

        row_index = self.gateway.add_product_to_order(matches[0].record_id)
        self.gateway.set_order_line(row_index, item)
        self.evidence.record(
            "item_completed", source_index=source_index, row_index=row_index, sku=item.sku
        )

    def _ensure_vat(self, vat_percent: Decimal) -> str:
        label = f"VAT {_decimal_label(vat_percent)}%"
        candidates = self.gateway.search_vats(label)
        exact = [
            candidate
            for candidate in candidates
            if normalize_text(candidate.name) == normalize_text(label)
            and candidate.value == vat_percent
            and normalize_text(candidate.e_invoice_code) == "s"
        ]
        if len(candidates) > 1:
            raise ManualReviewRequired(
                f"multiple or conflicting VAT records match {label!r}"
            )
        if len(exact) == 1:
            return exact[0].record_id
        if candidates:
            raise ManualReviewRequired(f"VAT record {label!r} exists with conflicting settings")
        return self.gateway.create_vat(
            VatCandidate(
                record_id="",
                name=label,
                description=label,
                value=vat_percent,
                e_invoice_code="S",
            )
        )

    def _verify_order_editor(self, expected: OrderInput) -> None:
        actual = self.gateway.read_order_snapshot()
        if actual.order_date != expected.order_date:
            raise VerificationError("Order date was not persisted in the editor")
        if normalize_text(actual.external_reference) != normalize_text(
            expected.external_reference
        ):
            raise VerificationError("Order external reference differs from source")
        if len(actual.items) != len(expected.items):
            raise VerificationError(
                f"Order has {len(actual.items)} lines, expected {len(expected.items)}"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual.items, expected.items, strict=True)
        ):
            if actual_item != expected_item:
                raise VerificationError(
                    f"Order item {index} differs from the extracted source"
                )
        for field in ("total_net", "total_vat", "total_gross"):
            if not within_tolerance(
                getattr(actual.totals, field), getattr(expected.totals, field)
            ):
                raise VerificationError(f"Order {field} differs from source")

    def _verify_linked_invoice(self, expected: OrderInput) -> None:
        actual = self.gateway.read_invoice_snapshot()
        if normalize_text(actual.external_reference) != normalize_text(
            expected.external_reference
        ):
            raise VerificationError("Invoice external reference was not copied from Order")
        if actual.order_date != expected.order_date:
            raise VerificationError("Invoice Order Date was not copied")
        if len(actual.items) != len(expected.items):
            raise VerificationError("Invoice line count differs from Order")
        if not addresses_match(
            actual.debtor.billing_address, expected.debtor.billing_address
        ) or not addresses_match(
            actual.debtor.effective_delivery_address,
            expected.debtor.effective_delivery_address,
        ):
            raise VerificationError("Invoice addresses differ from Order")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual.items, expected.items, strict=True)
        ):
            if actual_item != expected_item:
                raise VerificationError(
                    f"Invoice item {index} differs from the source Order"
                )
        for field in ("total_net", "total_vat", "total_gross"):
            if not within_tolerance(
                getattr(actual.totals, field), getattr(expected.totals, field)
            ):
                raise VerificationError(f"Invoice {field} differs from Order")

    @staticmethod
    def _verify_document_identity(
        expected: DocumentRecord, actual: DocumentRecord
    ) -> None:
        if expected.record_id != actual.record_id or expected.number != actual.number:
            raise VerificationError("verified document identity differs from saved document")
        for field in (
            "document_type",
            "date",
            "external_reference",
            "state",
            "transaction_id",
        ):
            if getattr(expected, field) != getattr(actual, field):
                raise VerificationError(
                    f"verified document {field} differs from saved document"
                )
        if not within_tolerance(expected.total, actual.total):
            raise VerificationError("verified document total differs from saved document")

    def _transition(self, state: WorkflowState, **details) -> None:
        self.state = state
        self.evidence.record("state_transition", state=state.value, **details)
        screenshot = self.evidence.screenshot_path(state.value.lower())
        try:
            captured = self.gateway.capture_screenshot(screenshot)
        except Exception as exc:  # evidence failure must never hide business result
            captured = False
            self.evidence.record("screenshot_failed", state=state.value, error=str(exc))
        self.checkpoint_store.save(
            {
                "workflow_id": self.workflow_id,
                "state": state.value,
                "external_reference": self.external_reference,
                "order_number": self.order_number,
                "invoice_number": self.invoice_number,
                "order_document": (
                    self.order_document.model_dump(mode="json")
                    if self.order_document
                    else None
                ),
                "invoice_document": (
                    self.invoice_document.model_dump(mode="json")
                    if self.invoice_document
                    else None
                ),
                "screenshot": str(screenshot) if captured else None,
                "details": details,
            }
        )


def _decimal_label(value: Decimal) -> str:
    return format(value.normalize(), "f")
