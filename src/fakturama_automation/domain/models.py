from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Money = Annotated[Decimal, Field(max_digits=14, decimal_places=4)]
Percentage = Annotated[Decimal, Field(ge=0, le=100, max_digits=7, decimal_places=4)]
Quantity = Annotated[Decimal, Field(gt=0, max_digits=14, decimal_places=4)]


class PaymentStatus(StrEnum):
    PAID = "PAID"
    UNPAID = "UNPAID"
    OVERDUE = "OVERDUE"
    PARTIALLY_PAID = "PARTIALLY PAID"
    REFUNDED = "REFUNDED"


class Address(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    street: str = Field(min_length=1)
    zip: str = Field(min_length=1)
    city: str = Field(min_length=1)
    country: str = Field(min_length=1)
    email: str | None = None
    telephone: str | None = None
    additional_name: str | None = None
    address_specification: str | None = None
    district: str | None = None


class Debtor(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    company: str = ""
    first_name: str = ""
    last_name: str = ""
    alias: str = ""
    salutation: str | None = None
    billing_address: Address
    delivery_address: Address | None = None

    @model_validator(mode="after")
    def require_identity(self) -> "Debtor":
        if not self.company and not (self.first_name or self.last_name):
            raise ValueError("debtor requires a company or contact name")
        return self

    @property
    def effective_delivery_address(self) -> Address:
        return self.delivery_address or self.billing_address


class Payment(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    method: str = Field(min_length=1)
    status: PaymentStatus
    payment_date: date | None = None


class OrderItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    sku: str = Field(min_length=1)
    description: str = Field(min_length=1)
    quantity: Quantity
    unit_net_price: Money = Field(ge=0)
    vat_percent: Percentage
    discount_percent: Percentage = Decimal("0")
    source_total: Money = Field(ge=0)

    @field_validator(
        "quantity",
        "unit_net_price",
        "vat_percent",
        "discount_percent",
        "source_total",
        mode="before",
    )
    @classmethod
    def normalize_decimal(cls, value: Any) -> Decimal:
        if isinstance(value, str):
            cleaned = value.strip().replace(" ", "")
            if "," in cleaned and "." not in cleaned:
                cleaned = cleaned.replace(",", ".")
            value = cleaned
        return Decimal(str(value))


class OrderTotals(BaseModel):
    total_net: Money = Field(ge=0)
    total_vat: Money = Field(ge=0)
    total_gross: Money = Field(ge=0)

    @field_validator("total_net", "total_vat", "total_gross", mode="before")
    @classmethod
    def normalize_decimal(cls, value: Any) -> Decimal:
        if isinstance(value, str):
            cleaned = value.strip().replace(" ", "")
            if "," in cleaned and "." not in cleaned:
                cleaned = cleaned.replace(",", ".")
            value = cleaned
        return Decimal(str(value))


class FieldEvidence(BaseModel):
    source_text: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: tuple[int, int, int, int] | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    page: int = Field(default=1, ge=1)


class OrderInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    order_date: date
    external_reference: str = Field(min_length=1)
    debtor: Debtor
    payment: Payment
    items: list[OrderItem] = Field(min_length=1)
    totals: OrderTotals
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    evidence: dict[str, FieldEvidence] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, value: str) -> str:
        return value.upper()


class DebtorCandidate(BaseModel):
    record_id: str
    company: str = ""
    first_name: str = ""
    last_name: str = ""
    billing_address: Address
    delivery_address: Address | None = None


class ProductCandidate(BaseModel):
    record_id: str
    sku: str
    name: str
    vat_percent: Decimal
    gross_price: Decimal


class VatCandidate(BaseModel):
    record_id: str
    name: str
    description: str
    value: Decimal
    e_invoice_code: str


class PaymentMethodCandidate(BaseModel):
    record_id: str
    name: str
    description: str
    payment_code: str
    cash_discount: Decimal = Decimal("0")
    discount_days: int = 0
    net_days: int = 0


class DocumentRecord(BaseModel):
    record_id: str
    document_type: str
    number: str
    date: date
    external_reference: str
    state: str
    total: Decimal
    transaction_id: str


class OrderSnapshot(BaseModel):
    number: str
    order_date: date
    external_reference: str
    debtor: Debtor
    payment_method: str
    items: list[OrderItem]
    totals: OrderTotals
    state: str


class InvoiceSnapshot(BaseModel):
    number: str
    invoice_date: date
    service_date: date
    order_date: date
    external_reference: str
    debtor: Debtor
    payment: Payment
    items: list[OrderItem]
    totals: OrderTotals
    state: str
