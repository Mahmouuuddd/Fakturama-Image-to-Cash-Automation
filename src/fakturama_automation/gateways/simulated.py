from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from fakturama_automation.domain.errors import VerificationError
from fakturama_automation.domain.matching import main_address_only
from fakturama_automation.domain.models import (
    Debtor,
    DebtorCandidate,
    DocumentRecord,
    InvoiceSnapshot,
    OrderInput,
    OrderItem,
    OrderSnapshot,
    Payment,
    PaymentMethodCandidate,
    PaymentStatus,
    ProductCandidate,
    VatCandidate,
)


class SimulatedFakturamaGateway:
    """Faithful in-memory adapter for workflow and recovery testing."""

    def __init__(
        self,
        *,
        debtors: list[tuple[DebtorCandidate, str]] | None = None,
        payment_methods: list[PaymentMethodCandidate] | None = None,
        vats: list[VatCandidate] | None = None,
        products: list[ProductCandidate] | None = None,
    ) -> None:
        self.debtors = {
            candidate.record_id: (candidate, payment_method)
            for candidate, payment_method in (debtors or [])
        }
        self.payment_methods = {
            candidate.record_id: candidate for candidate in (payment_methods or [])
        }
        self.vats = {candidate.record_id: candidate for candidate in (vats or [])}
        self.products = {
            candidate.record_id: candidate for candidate in (products or [])
        }
        self.documents: dict[str, DocumentRecord] = {}
        self._order_input: OrderInput | None = None
        self._selected_debtor_id: str | None = None
        self._order_lines: list[OrderItem | None] = []
        self._order_number: str | None = None
        self._invoice_number: str | None = None
        self._invoice_payment: Payment | None = None
        self._transaction_id: str | None = None
        self._pending_debtor: Debtor | None = None
        self._pending_payment_method: str | None = None
        self.events: list[str] = []

    def preflight(self) -> None:
        return None

    def open_new_order(self) -> str:
        self.events.append("open_new_order")
        self._order_number = f"ORD-{len(self.documents) + 1:05d}"
        self._transaction_id = uuid4().hex
        return self._order_number

    def set_order_header(self, order: OrderInput) -> None:
        if self._order_number is None:
            raise RuntimeError("no Order editor is open")
        self._order_input = deepcopy(order)

    def search_debtors(self, company_or_name: str) -> list[DebtorCandidate]:
        query = company_or_name.casefold().strip()
        return [
            deepcopy(candidate)
            for candidate, _ in self.debtors.values()
            if query in " ".join(
                (candidate.company, candidate.first_name, candidate.last_name)
            ).casefold()
        ]

    def select_debtor(self, record_id: str) -> None:
        if record_id not in self.debtors:
            raise KeyError(record_id)
        self._selected_debtor_id = record_id

    def cancel_active_dialog(self) -> None:
        return None

    def read_selected_debtor(self) -> Debtor:
        if self._selected_debtor_id is None:
            raise RuntimeError("no Debtor is selected")
        candidate, _ = self.debtors[self._selected_debtor_id]
        return Debtor(
            company=candidate.company,
            first_name=candidate.first_name,
            last_name=candidate.last_name,
            billing_address=deepcopy(candidate.billing_address),
            delivery_address=deepcopy(candidate.delivery_address),
        )

    def search_payment_methods(self, name: str) -> list[PaymentMethodCandidate]:
        query = name.casefold().strip()
        return [
            deepcopy(candidate)
            for candidate in self.payment_methods.values()
            if query in candidate.name.casefold()
        ]

    def create_payment_method(self, payment: PaymentMethodCandidate) -> str:
        self.events.append("create_payment_method")
        record_id = f"payment-{len(self.payment_methods) + 1}"
        saved = payment.model_copy(update={"record_id": record_id})
        self.payment_methods[record_id] = saved
        return record_id

    def open_new_debtor(self, debtor: Debtor) -> None:
        self.events.append("open_new_debtor")
        self._pending_debtor = deepcopy(main_address_only(debtor))
        self._pending_payment_method = None

    def discard_and_reopen_debtor(self, debtor: Debtor) -> None:
        self.events.append("discard_and_reopen_debtor")
        self._pending_debtor = None
        self._pending_payment_method = None
        self.open_new_debtor(debtor)

    def select_debtor_payment_method(self, payment_method: str) -> bool:
        if self._pending_debtor is None:
            raise RuntimeError("no Debtor editor is open")
        exact = [
            candidate
            for candidate in self.payment_methods.values()
            if candidate.name.casefold().strip() == payment_method.casefold().strip()
        ]
        if len(exact) != 1:
            return False
        self.events.append("select_debtor_payment_method")
        self._pending_payment_method = exact[0].name
        return True

    def save_debtor(self) -> str:
        if self._pending_debtor is None or self._pending_payment_method is None:
            raise RuntimeError("Debtor editor is incomplete")
        self.events.append("save_debtor")
        debtor = self._pending_debtor
        payment_method = self._pending_payment_method
        record_id = f"debtor-{len(self.debtors) + 1}"
        candidate = DebtorCandidate(
            record_id=record_id,
            company=debtor.company,
            first_name=debtor.first_name,
            last_name=debtor.last_name,
            billing_address=deepcopy(debtor.billing_address),
            delivery_address=deepcopy(debtor.delivery_address),
        )
        self.debtors[record_id] = (candidate, payment_method)
        self._pending_debtor = None
        self._pending_payment_method = None
        return record_id

    def search_vats(self, name: str) -> list[VatCandidate]:
        query = name.casefold().strip()
        return [
            deepcopy(candidate)
            for candidate in self.vats.values()
            if query in candidate.name.casefold()
        ]

    def create_vat(self, vat: VatCandidate) -> str:
        self.events.append("create_vat")
        record_id = f"vat-{len(self.vats) + 1}"
        self.vats[record_id] = vat.model_copy(update={"record_id": record_id})
        return record_id

    def search_products(self, sku: str) -> list[ProductCandidate]:
        query = sku.casefold().strip()
        return [
            deepcopy(candidate)
            for candidate in self.products.values()
            if query in candidate.sku.casefold()
        ]

    def create_product(self, product: ProductCandidate) -> str:
        self.events.append("create_product")
        record_id = f"product-{len(self.products) + 1}"
        self.products[record_id] = product.model_copy(update={"record_id": record_id})
        return record_id

    def add_product_to_order(self, record_id: str) -> int:
        if record_id not in self.products:
            raise KeyError(record_id)
        self._order_lines.append(None)
        return len(self._order_lines) - 1

    def set_order_line(self, row_index: int, item: OrderItem) -> None:
        self.events.append(f"set_order_line:{item.sku}")
        self._order_lines[row_index] = deepcopy(item)

    def complete_order(self) -> None:
        if self._order_input is None:
            raise RuntimeError("no Order editor is open")
        self.events.append("complete_order")

    def read_order_snapshot(self) -> OrderSnapshot:
        order = self._require_order()
        if self._selected_debtor_id is None or any(line is None for line in self._order_lines):
            raise RuntimeError("Order is incomplete")
        _, payment_method = self.debtors[self._selected_debtor_id]
        return OrderSnapshot(
            number=self._order_number,
            order_date=order.order_date,
            external_reference=order.external_reference,
            debtor=self.read_selected_debtor(),
            payment_method=payment_method,
            items=deepcopy(self._order_lines),
            totals=deepcopy(order.totals),
            state="open",
        )

    def save_order(self) -> DocumentRecord:
        self.events.append("save_order")
        snapshot = self.read_order_snapshot()
        record_id = f"document-{len(self.documents) + 1}"
        document = DocumentRecord(
            record_id=record_id,
            document_type="Order",
            number=snapshot.number,
            date=snapshot.order_date,
            external_reference=snapshot.external_reference,
            state="open",
            total=snapshot.totals.total_gross,
            transaction_id=self._transaction_id,
        )
        self.documents[record_id] = document
        return deepcopy(document)

    def verify_document(self, expected: DocumentRecord) -> DocumentRecord:
        self.events.append(f"verify_document:{expected.document_type}")
        if expected.record_id not in self.documents:
            raise VerificationError(f"document {expected.record_id} was not persisted")
        return deepcopy(self.documents[expected.record_id])

    def create_follow_up_invoice(self) -> str:
        self.events.append("create_follow_up_invoice")
        if not any(doc.document_type == "Order" for doc in self.documents.values()):
            raise RuntimeError("Order must be saved before creating an Invoice")
        self._invoice_number = f"INV-{len(self.documents) + 1:05d}"
        self._require_order()
        if self._selected_debtor_id is None:
            raise RuntimeError("Invoice requires a selected Debtor")
        _, selected_payment_method = self.debtors[self._selected_debtor_id]
        self._invoice_payment = Payment(
            method=selected_payment_method,
            status=PaymentStatus.UNPAID,
            payment_date=None,
        )
        return self._invoice_number

    def read_invoice_snapshot(self) -> InvoiceSnapshot:
        self.events.append("read_invoice_snapshot")
        if self._invoice_number is None or self._invoice_payment is None:
            raise RuntimeError("no linked Invoice editor is open")
        order = self._require_order()
        return InvoiceSnapshot(
            number=self._invoice_number,
            invoice_date=date.today(),
            service_date=date.today(),
            order_date=order.order_date,
            external_reference=order.external_reference,
            debtor=self.read_selected_debtor(),
            payment=deepcopy(self._invoice_payment),
            items=deepcopy(self._order_lines),
            totals=deepcopy(order.totals),
            state="unpaid",
        )

    def set_invoice_payment(self, payment: Payment, invoice_total: Decimal) -> None:
        self.events.append("set_invoice_payment")
        order = self._require_order()
        if invoice_total != order.totals.total_gross:
            raise VerificationError("payment value differs from Invoice total")
        self._invoice_payment = deepcopy(payment)

    def save_invoice(self) -> DocumentRecord:
        self.events.append("save_invoice")
        snapshot = self.read_invoice_snapshot()
        record_id = f"document-{len(self.documents) + 1}"
        document = DocumentRecord(
            record_id=record_id,
            document_type="Invoice",
            number=snapshot.number,
            date=snapshot.invoice_date,
            external_reference=snapshot.external_reference,
            state="paid" if snapshot.payment.status is PaymentStatus.PAID else "unpaid",
            total=snapshot.totals.total_gross,
            transaction_id=self._transaction_id,
        )
        self.documents[record_id] = document
        return deepcopy(document)

    def capture_screenshot(self, path: Path) -> bool:
        return False

    def _require_order(self) -> OrderInput:
        if self._order_input is None or self._order_number is None:
            raise RuntimeError("no Order editor is open")
        return self._order_input
