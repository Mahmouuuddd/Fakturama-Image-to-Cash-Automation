from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .ocr import OcrResult


class EvidencePage(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[int, int, int, int]
    page: int = Field(default=1, ge=1)
    reading_order: int = Field(ge=0)
    variant: str = "primary"
    model: str = "unknown"


class EvidenceRegion(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    bbox: tuple[int, int, int, int]
    text: str = ""
    page: int = Field(default=1, ge=1)


class EvidenceTableColumn(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    header_ids: tuple[str, ...]
    x_range: tuple[int, int]


class EvidenceTableCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    text: str
    evidence_ids: tuple[str, ...]


class EvidenceTableRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    y_range: tuple[int, int]
    cells: tuple[EvidenceTableCell, ...]


class EvidenceTable(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1)
    header_ids: tuple[str, ...]
    columns: tuple[EvidenceTableColumn, ...]
    rows: tuple[EvidenceTableRow, ...]


class EvidenceDocument(BaseModel):
    """Immutable, locally-authored bridge between OCR and semantic parsing."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source_path: str
    source_sha256: str
    pages: tuple[EvidencePage, ...]
    spans: tuple[EvidenceSpan, ...]
    regions: tuple[EvidenceRegion, ...] = ()
    tables: tuple[EvidenceTable, ...] = ()
    preprocessing_warnings: tuple[str, ...] = ()

    def span_index(self) -> dict[str, EvidenceSpan]:
        return {span.id: span for span in self.spans}

    def prompt_payload(self) -> dict:
        """Return the compact semantic payload sent to a remote text model.

        Full provider/audit metadata remains in ``evidence.json``. Reading order
        is the array order, and the page is omitted from spans for one-page input.
        """
        multiple_pages = len(self.pages) > 1
        span_columns = ["id", "text", "confidence", "bbox"]
        if multiple_pages:
            span_columns.append("page")
        spans = []
        for span in self.spans:
            row: list[object] = [
                span.id,
                span.text,
                round(span.confidence, 3),
                list(span.bbox),
            ]
            if multiple_pages:
                row.append(span.page)
            spans.append(row)
        payload = {
            "pages": [[page.page, page.width, page.height] for page in self.pages],
            "page_columns": ["page", "width", "height"],
            "span_columns": span_columns,
            "spans": spans,
        }
        if self.regions:
            payload["regions"] = [
                [region.kind, region.text, list(region.bbox), region.page]
                for region in self.regions
            ]
            payload["region_columns"] = ["kind", "text", "bbox", "page"]
        if self.tables:
            item_tables = []
            global_row_offset = 0
            for table in sorted(self.tables, key=lambda item: item.page):
                item_tables.append({
                    "page": table.page,
                    "columns": [
                        [column.role, list(column.header_ids), list(column.x_range)]
                        for column in table.columns
                    ],
                    "column_columns": ["role", "header_ids", "x_range"],
                    "rows": [
                        [
                            global_row_offset + row.index,
                            [
                                [cell.role, list(cell.evidence_ids), cell.text]
                                for cell in row.cells
                            ],
                        ]
                        for row in table.rows
                    ],
                    "row_columns": ["zero_based_item_index", "cells"],
                    "cell_columns": ["role", "evidence_ids", "text"],
                })
                global_row_offset += len(table.rows)
            payload["item_tables"] = item_tables
        return payload

    def table_cell_index(self) -> dict[str, tuple[int, str]]:
        index: dict[str, tuple[int, str]] = {}
        global_row_offset = 0
        for table in sorted(self.tables, key=lambda item: item.page):
            for row in table.rows:
                for cell in row.cells:
                    for evidence_id in cell.evidence_ids:
                        index[evidence_id] = (global_row_offset + row.index, cell.role)
            global_row_offset += len(table.rows)
        return index


class DraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedField(DraftModel):
    """One LLM-interpreted value and the local OCR spans claimed to support it."""

    value: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    ambiguity: str | None = None


class DraftAddress(DraftModel):
    street: ExtractedField
    zip: ExtractedField
    city: ExtractedField
    country: ExtractedField
    email: ExtractedField = Field(default_factory=ExtractedField)
    telephone: ExtractedField = Field(default_factory=ExtractedField)
    additional_name: ExtractedField = Field(default_factory=ExtractedField)
    address_specification: ExtractedField = Field(default_factory=ExtractedField)
    district: ExtractedField = Field(default_factory=ExtractedField)


class DraftDebtor(DraftModel):
    company: ExtractedField = Field(default_factory=ExtractedField)
    first_name: ExtractedField = Field(default_factory=ExtractedField)
    last_name: ExtractedField = Field(default_factory=ExtractedField)
    alias: ExtractedField = Field(default_factory=ExtractedField)
    salutation: ExtractedField = Field(default_factory=ExtractedField)
    billing_address: DraftAddress
    delivery_address: DraftAddress | None = None


class DraftPayment(DraftModel):
    method: ExtractedField
    status: ExtractedField
    payment_date: ExtractedField = Field(default_factory=ExtractedField)


class DraftItem(DraftModel):
    sku: ExtractedField
    description: ExtractedField
    quantity: ExtractedField
    unit_net_price: ExtractedField
    vat_percent: ExtractedField
    discount_percent: ExtractedField
    source_total: ExtractedField


class DraftTotals(DraftModel):
    total_net: ExtractedField
    total_vat: ExtractedField
    total_gross: ExtractedField


class ExtractionDraft(DraftModel):
    """Nullable semantic extraction. It is not yet authorized business input."""

    order_date: ExtractedField
    external_reference: ExtractedField
    currency: ExtractedField
    debtor: DraftDebtor
    payment: DraftPayment
    items: list[DraftItem] = Field(default_factory=list)
    totals: DraftTotals


class FieldClaim(DraftModel):
    """Compact LLM output row; only local code may interpret its path."""

    path: str
    value: str | None
    evidence_ids: list[str]
    ambiguity: str | None


class CompactExtractionClaims(DraftModel):
    item_count: int = Field(ge=0, le=100)
    delivery_address_present: bool
    claims: list[FieldClaim]


_ROOT_CLAIM_PATHS = {
    "order_date",
    "external_reference",
    "currency",
    "debtor.company",
    "debtor.first_name",
    "debtor.last_name",
    "debtor.alias",
    "debtor.salutation",
    "payment.method",
    "payment.status",
    "payment.payment_date",
    "totals.total_net",
    "totals.total_vat",
    "totals.total_gross",
}
_ADDRESS_FIELDS = {
    "street",
    "zip",
    "city",
    "country",
    "email",
    "telephone",
    "additional_name",
    "address_specification",
    "district",
}
_ITEM_FIELDS = {
    "sku",
    "description",
    "quantity",
    "unit_net_price",
    "vat_percent",
    "discount_percent",
    "source_total",
}
_CLAIM_PATH_ALIASES = {
    "debtor.email": "debtor.billing_address.email",
    "debtor.telephone": "debtor.billing_address.telephone",
}


def claims_to_draft(response: CompactExtractionClaims) -> ExtractionDraft:
    """Expand allowlisted field claims into the nullable nested draft model."""

    def empty_field() -> dict:
        return {"value": None, "evidence_ids": [], "ambiguity": None}

    def empty_address() -> dict:
        return {name: empty_field() for name in _ADDRESS_FIELDS}

    raw = {
        "order_date": empty_field(),
        "external_reference": empty_field(),
        "currency": empty_field(),
        "debtor": {
            "company": empty_field(),
            "first_name": empty_field(),
            "last_name": empty_field(),
            "alias": empty_field(),
            "salutation": empty_field(),
            "billing_address": empty_address(),
            "delivery_address": empty_address()
            if response.delivery_address_present
            else None,
        },
        "payment": {
            "method": empty_field(),
            "status": empty_field(),
            "payment_date": empty_field(),
        },
        "items": [
            {name: empty_field() for name in _ITEM_FIELDS}
            for _ in range(response.item_count)
        ],
        "totals": {
            "total_net": empty_field(),
            "total_vat": empty_field(),
            "total_gross": empty_field(),
        },
    }

    seen: set[str] = set()
    for claim in response.claims:
        path = _CLAIM_PATH_ALIASES.get(claim.path, claim.path)
        if path in seen:
            raise ValueError(f"duplicate extraction claim path: {path}")
        seen.add(path)
        _validate_claim_path(path, response)
        _set_claim(raw, path, claim.model_dump(exclude={"path"}))
    return ExtractionDraft.model_validate(raw)


def _validate_claim_path(path: str, response: CompactExtractionClaims) -> None:
    if path in _ROOT_CLAIM_PATHS:
        return
    address = re.fullmatch(
        r"debtor\.(billing_address|delivery_address)\.([a-z_]+)", path
    )
    if address:
        address_kind, field = address.groups()
        if field not in _ADDRESS_FIELDS:
            raise ValueError(f"unsupported extraction claim path: {path}")
        if address_kind == "delivery_address" and not response.delivery_address_present:
            raise ValueError("delivery-address claim supplied while address is absent")
        return
    item = re.fullmatch(r"items\[(\d+)]\.([a-z_]+)", path)
    if item:
        index, field = int(item.group(1)), item.group(2)
        if index >= response.item_count or field not in _ITEM_FIELDS:
            raise ValueError(f"unsupported extraction claim path: {path}")
        return
    raise ValueError(f"unsupported extraction claim path: {path}")


def _set_claim(raw: dict, path: str, value: dict) -> None:
    current: object = raw
    parts = re.findall(r"[^.\[\]]+", path)
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else current[part]
    current[parts[-1]] = value


def build_evidence_document(image_path: Path, ocr: OcrResult) -> EvidenceDocument:
    source_path = image_path.resolve()
    source_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    ordered = sorted(
        ocr.lines,
        key=lambda line: (
            line.page,
            line.bounding_box[1],
            line.bounding_box[0],
            line.bounding_box[3],
            line.bounding_box[2],
        ),
    )
    spans = tuple(
        EvidenceSpan(
            id=f"ocr_{index:04d}",
            text=line.text,
            confidence=line.confidence,
            bbox=line.bounding_box,
            page=line.page,
            reading_order=index - 1,
            variant=line.variant,
            model=line.model,
        )
        for index, line in enumerate(ordered, start=1)
    )
    if ocr.quality:
        page_width = ocr.quality.width
        page_height = ocr.quality.height
        warnings = ocr.quality.issues
    else:
        page_width = max((line.bounding_box[2] for line in ordered), default=1)
        page_height = max((line.bounding_box[3] for line in ordered), default=1)
        warnings = ()
    regions = tuple(
        EvidenceRegion(
            kind=region.kind,
            bbox=region.bounding_box,
            text=region.text,
        )
        for region in ocr.regions
    )
    from .tables import infer_item_tables

    tables = infer_item_tables(
        spans,
        (EvidencePage(page=1, width=page_width, height=page_height),),
    )
    return EvidenceDocument(
        document_id=f"sha256:{source_hash}",
        source_path=str(source_path),
        source_sha256=source_hash,
        pages=(EvidencePage(page=1, width=page_width, height=page_height),),
        spans=spans,
        regions=regions,
        tables=tables,
        preprocessing_warnings=warnings,
    )
