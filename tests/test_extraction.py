import json
from decimal import Decimal
from pathlib import Path

import pytest

from fakturama_automation.domain.models import FieldEvidence, OrderInput
from fakturama_automation.domain.numbers import (
    decimal_values_in_text,
    parse_decimal_text,
)
from fakturama_automation.domain.validation import (
    Severity,
    ValidationIssue,
    ValidationReport,
)
from fakturama_automation.extraction.ocr import OcrLine, OcrResult
from fakturama_automation.extraction.evidence import (
    CompactExtractionClaims,
    DraftAddress,
    EvidenceDocument,
    EvidencePage,
    EvidenceSpan,
    EvidenceTable,
    EvidenceTableCell,
    EvidenceTableColumn,
    EvidenceTableRow,
    ExtractionDraft,
    build_evidence_document,
    claims_to_draft,
)
from fakturama_automation.extraction.preprocessing import OpenCvPreprocessor
from fakturama_automation.extraction.pipeline import (
    ImageOrderExtractor,
    _evidence_text_matches,
    _validate_table_cell_claim,
    _value_is_supported,
    ground_extraction_draft,
)
import fakturama_automation.extraction.parser as parser_module
from fakturama_automation.extraction.parser import (
    CompatibleChatConfig,
    CompatibleChatOrderParser,
    _parse_json_object,
    _strict_json_schema,
    _supports_strict_json_schema,
)
from fakturama_automation.extraction.verification import (
    normalize_delivery_presence,
    normalize_optional_placeholders,
    normalize_proven_ambiguities,
    trusted_address_claims,
    trusted_company_claims,
    trusted_item_claims,
    trusted_total_claims,
)
from fakturama_automation.infrastructure.review import write_review_packet
from PIL import Image


OCR_VALUES = [
    ("order_date", "Order date: 2026-08-14", "2026-08-14"),
    ("reference", "Reference: WEB-2026-0001", "WEB-2026-0001"),
    ("currency", "Currency: EUR", "EUR"),
    ("company", "Müller GmbH", "Müller GmbH"),
    ("street", "Königstraße 1", "Königstraße 1"),
    ("zip", "10117", "10117"),
    ("city", "Berlin", "Berlin"),
    ("country", "Deutschland", "Deutschland"),
    ("method", "Bank transfer", "Bank transfer"),
    ("status", "Status: UNPAID", "UNPAID"),
    ("sku", "CH-1", "CH-1"),
    ("description", "Stainless chair", "Stainless chair"),
    ("quantity", "2", "2"),
    ("unit", "10.00", "10.00"),
    ("vat", "19%", "19"),
    ("discount", "0%", "0"),
    ("line_total", "20.00", "20.00"),
    ("net", "Total net 20.00", "20.00"),
    ("total_vat", "Total VAT 3.80", "3.80"),
    ("gross", "Total gross 23.80 EUR", "23.80"),
]


def test_trusted_totals_use_explicit_labels_and_relative_geometry() -> None:
    spans = (
        EvidenceSpan(id="n-label", text="NET TOTAL", confidence=0.99, bbox=(100, 400, 200, 430), reading_order=0),
        EvidenceSpan(id="v-label", text="VAT TOTAL", confidence=0.99, bbox=(300, 400, 400, 430), reading_order=1),
        EvidenceSpan(id="g-label", text="GROSS TOTAL", confidence=0.99, bbox=(500, 400, 620, 430), reading_order=2),
        EvidenceSpan(id="n-value", text="EUR 570.00", confidence=0.99, bbox=(90, 450, 210, 480), reading_order=3),
        EvidenceSpan(id="v-value", text="EUR 108.30", confidence=0.99, bbox=(290, 450, 410, 480), reading_order=4),
        EvidenceSpan(id="g-value", text="EUR 678.30", confidence=0.99, bbox=(500, 450, 620, 480), reading_order=5),
        EvidenceSpan(id="noise", text="250.00", confidence=0.99, bbox=(10, 200, 80, 230), reading_order=6),
    )
    document = EvidenceDocument(
        document_id="totals-test",
        source_path="order.png",
        source_sha256="0" * 64,
        pages=(EvidencePage(page=1, width=800, height=1000),),
        spans=spans,
    )

    claims = {claim.path: claim for claim in trusted_total_claims(document)}

    assert claims["totals.total_net"].value == "570.00"
    assert claims["totals.total_vat"].value == "108.30"
    assert claims["totals.total_gross"].value == "678.30"
    assert claims["totals.total_gross"].evidence_ids == ["g-label", "g-value"]


def test_trusted_addresses_use_explicit_side_by_side_labels_and_geometry() -> None:
    values = (
        ("billing-label", "Billing Address", (80, 100, 280, 130)),
        ("delivery-label", "Delivery Address", (620, 100, 850, 130)),
        ("billing-name", "Northstar Office GmbH", (80, 145, 310, 170)),
        ("delivery-name", "Northstar Warehouse", (620, 145, 850, 170)),
        ("delivery-street", "Beusselstrasse 44", (620, 185, 830, 210)),
        ("billing-street", "Friedrichstrasse 88", (80, 185, 300, 210)),
        ("billing-postal", "10117 Berlin", (80, 225, 220, 250)),
        ("delivery-postal", "10553 Berlin", (620, 225, 760, 250)),
        ("billing-country", "Germany", (80, 265, 180, 290)),
        ("delivery-country", "Germany", (620, 265, 720, 290)),
        ("payment", "PAYMENT", (60, 340, 180, 370)),
    )
    document = EvidenceDocument(
        document_id="address-test",
        source_path="order.png",
        source_sha256="0" * 64,
        pages=(EvidencePage(page=1, width=1000, height=600),),
        spans=tuple(
            EvidenceSpan(
                id=evidence_id,
                text=text,
                confidence=0.99,
                bbox=bbox,
                reading_order=index,
            )
            for index, (evidence_id, text, bbox) in enumerate(values)
        ),
    )

    delivery_present, claims = trusted_address_claims(document)
    claims_by_path = {claim.path: claim for claim in claims}

    assert delivery_present is True
    assert claims_by_path["debtor.billing_address.street"].value == "Friedrichstrasse 88"
    assert claims_by_path["debtor.billing_address.zip"].value == "10117"
    assert claims_by_path["debtor.billing_address.city"].value == "Berlin"
    assert claims_by_path["debtor.billing_address.country"].value == "Germany"
    assert claims_by_path["debtor.delivery_address.street"].value == "Beusselstrasse 44"
    assert claims_by_path["debtor.delivery_address.zip"].evidence_ids == ["delivery-postal"]


def test_trusted_addresses_reject_an_ambiguous_postal_column() -> None:
    values = (
        ("billing-label", "Billing Address", (80, 100, 280, 130)),
        ("delivery-label", "Delivery Address", (620, 100, 850, 130)),
        ("billing-street", "One Street 1", (80, 145, 240, 170)),
        ("billing-postal-a", "10117 Berlin", (80, 185, 220, 210)),
        ("billing-postal-b", "20095 Hamburg", (80, 225, 240, 250)),
        ("billing-country", "Germany", (80, 265, 180, 290)),
        ("delivery-street", "Two Street 2", (620, 145, 780, 170)),
        ("delivery-postal", "10553 Berlin", (620, 185, 760, 210)),
        ("delivery-country", "Germany", (620, 225, 720, 250)),
    )
    document = EvidenceDocument(
        document_id="ambiguous-address-test",
        source_path="order.png",
        source_sha256="0" * 64,
        pages=(EvidencePage(page=1, width=1000, height=600),),
        spans=tuple(
            EvidenceSpan(
                id=evidence_id,
                text=text,
                confidence=0.99,
                bbox=bbox,
                reading_order=index,
            )
            for index, (evidence_id, text, bbox) in enumerate(values)
        ),
    )

    delivery_present, claims = trusted_address_claims(document)

    assert delivery_present is True
    assert claims == []


def test_trusted_company_uses_one_explicit_label_and_value() -> None:
    document = EvidenceDocument(
        document_id="company-test",
        source_path="order.png",
        source_sha256="0" * 64,
        pages=(EvidencePage(page=1, width=1000, height=600),),
        spans=(
            EvidenceSpan(
                id="company-label",
                text="COMPANY",
                confidence=0.99,
                bbox=(80, 100, 220, 130),
                reading_order=0,
            ),
            EvidenceSpan(
                id="company-value",
                text="Northstar Office GmbH",
                confidence=0.99,
                bbox=(80, 145, 330, 175),
                reading_order=1,
            ),
            EvidenceSpan(
                id="contact-label",
                text="CONTACT NAME",
                confidence=0.99,
                bbox=(620, 100, 820, 130),
                reading_order=2,
            ),
            EvidenceSpan(
                id="contact-value",
                text="Marta Klein",
                confidence=0.99,
                bbox=(620, 145, 780, 175),
                reading_order=3,
            ),
            EvidenceSpan(
                id="later-label",
                text="CUSTOMER ALIAS",
                confidence=0.99,
                bbox=(80, 225, 260, 255),
                reading_order=4,
            ),
        ),
    )

    claims = trusted_company_claims(document)

    assert len(claims) == 1
    assert claims[0].path == "debtor.company"
    assert claims[0].value == "Northstar Office GmbH"
    assert claims[0].evidence_ids == ["company-label", "company-value"]


def test_trusted_totals_refuse_duplicate_labels() -> None:
    document = EvidenceDocument(
        document_id="ambiguous-totals",
        source_path="order.png",
        source_sha256="0" * 64,
        pages=(EvidencePage(page=1, width=800, height=1000),),
        spans=(
            EvidenceSpan(id="label-1", text="TOTAL NET", confidence=0.99, bbox=(100, 400, 200, 430), reading_order=0),
            EvidenceSpan(id="label-2", text="SUBTOTAL", confidence=0.99, bbox=(100, 500, 200, 530), reading_order=1),
            EvidenceSpan(id="value-1", text="570.00", confidence=0.99, bbox=(100, 450, 200, 480), reading_order=2),
            EvidenceSpan(id="value-2", text="570.00", confidence=0.99, bbox=(100, 550, 200, 580), reading_order=3),
        ),
    )

    assert not trusted_total_claims(document)


def _field(name: str) -> dict:
    for index, (field_name, _, value) in enumerate(OCR_VALUES, start=1):
        if field_name == name:
            return {"value": value, "evidence_ids": [f"ocr_{index:04d}"]}
    raise KeyError(name)


def _valid_draft() -> ExtractionDraft:
    return ExtractionDraft.model_validate(
        {
            "order_date": _field("order_date"),
            "external_reference": _field("reference"),
            "currency": _field("currency"),
            "debtor": {
                "company": _field("company"),
                "billing_address": {
                    "street": _field("street"),
                    "zip": _field("zip"),
                    "city": _field("city"),
                    "country": _field("country"),
                },
            },
            "payment": {
                "method": _field("method"),
                "status": _field("status"),
            },
            "items": [
                {
                    "sku": _field("sku"),
                    "description": _field("description"),
                    "quantity": _field("quantity"),
                    "unit_net_price": _field("unit"),
                    "vat_percent": _field("vat"),
                    "discount_percent": _field("discount"),
                    "source_total": _field("line_total"),
                }
            ],
            "totals": {
                "total_net": _field("net"),
                "total_vat": _field("total_vat"),
                "total_gross": _field("gross"),
            },
        }
    )


class StubOcr:
    def recognize(self, image_path: Path) -> OcrResult:
        return OcrResult(
            lines=[
                OcrLine(
                    text=text,
                    confidence=0.82 if name == "order_date" else 0.96,
                    bounding_box=(10, index * 20, 200, index * 20 + 15),
                )
                for index, (name, text, _) in enumerate(OCR_VALUES, start=1)
            ]
        )


class StubParser:
    uses_image = False

    def __init__(self, draft: ExtractionDraft) -> None:
        self.draft = draft

    def parse(self, evidence) -> ExtractionDraft:
        return self.draft


def test_image_pipeline_grounds_evidence_ids_and_preserves_unicode(tmp_path: Path) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)

    outcome = ImageOrderExtractor(StubOcr(), StubParser(_valid_draft())).extract(image_path)

    assert outcome.order is not None
    assert outcome.order.debtor.company == "Müller GmbH"
    assert outcome.order.debtor.billing_address.street == "Königstraße 1"
    assert outcome.order.evidence["order_date"].confidence == 0.82
    assert outcome.order.evidence["order_date"].evidence_ids == ["ocr_0001"]
    assert outcome.order.evidence["order_date"].bounding_box == (10, 20, 200, 35)
    assert outcome.report.valid
    assert outcome.report.requires_review


def test_unknown_evidence_id_stops_order_authorization(tmp_path: Path) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.external_reference.evidence_ids = ["ocr_9999"]

    outcome = ImageOrderExtractor(StubOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is None
    assert any(issue.code == "UNKNOWN_EVIDENCE_ID" for issue in outcome.report.issues)


def test_identifier_case_cannot_be_silently_rewritten(tmp_path: Path) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.items[0].sku.value = "ch-1"

    outcome = ImageOrderExtractor(StubOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is None
    assert any(
        issue.code == "UNGROUNDED_VALUE" and issue.path == "items[0].sku"
        for issue in outcome.report.issues
    )


def test_malformed_decimal_becomes_review_issue_instead_of_crashing(tmp_path: Path) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.items[0].quantity.value = "pcs"

    outcome = ImageOrderExtractor(StubOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is None
    assert any(issue.code == "UNGROUNDED_VALUE" for issue in outcome.report.issues)


def test_percentage_symbols_are_safely_normalized_after_grounding(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.items[0].vat_percent.value = "19%"
    draft.items[0].discount_percent.value = "0%"

    outcome = ImageOrderExtractor(StubOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is not None
    assert outcome.order.items[0].vat_percent == 19
    assert outcome.order.items[0].discount_percent == 0


def test_combined_postal_city_claim_is_split_with_shared_evidence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.billing_address.zip.value = "10117 Berlin"
    draft.debtor.billing_address.city.value = None
    draft.debtor.billing_address.city.evidence_ids = []

    class CombinedAddressOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[5] = OcrLine(
                text="10117 Berlin",
                confidence=0.99,
                bounding_box=result.lines[5].bounding_box,
            )
            return result

    outcome = ImageOrderExtractor(CombinedAddressOcr(), StubParser(draft)).extract(
        image_path
    )

    assert outcome.order is not None
    assert outcome.order.debtor.billing_address.zip == "10117"
    assert outcome.order.debtor.billing_address.city == "Berlin"
    assert outcome.order.evidence["debtor.billing_address.zip"].evidence_ids == [
        "ocr_0006"
    ]
    assert outcome.order.evidence["debtor.billing_address.city"].evidence_ids == [
        "ocr_0006"
    ]
    assert outcome.report.valid


def test_postal_city_can_be_completed_from_its_cited_combined_ocr_span(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.billing_address.city.value = None
    draft.debtor.billing_address.city.evidence_ids = []

    class CombinedAddressOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[5] = OcrLine(
                text="10117 Berlin",
                confidence=0.99,
                bounding_box=result.lines[5].bounding_box,
            )
            return result

    outcome = ImageOrderExtractor(CombinedAddressOcr(), StubParser(draft)).extract(
        image_path
    )

    assert outcome.order is not None
    assert outcome.order.debtor.billing_address.zip == "10117"
    assert outcome.order.debtor.billing_address.city == "Berlin"
    assert outcome.report.valid


def test_suffix_postal_format_is_split_only_for_a_compatible_country(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.billing_address.zip.value = None
    draft.debtor.billing_address.zip.evidence_ids = []
    draft.debtor.billing_address.city.value = "New York, NY 10001"
    draft.debtor.billing_address.country.value = "United States"

    class UsAddressOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[6] = OcrLine(
                text="New York, NY 10001",
                confidence=0.99,
                bounding_box=result.lines[6].bounding_box,
            )
            result.lines[7] = OcrLine(
                text="United States",
                confidence=0.99,
                bounding_box=result.lines[7].bounding_box,
            )
            return result

    outcome = ImageOrderExtractor(UsAddressOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is not None
    assert outcome.order.debtor.billing_address.zip == "10001"
    assert outcome.order.debtor.billing_address.city == "New York, NY"
    assert outcome.report.valid


def test_ambiguous_combined_address_is_not_silently_repaired(tmp_path: Path) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.billing_address.zip.value = "10117 Berlin"
    draft.debtor.billing_address.city.value = None
    draft.debtor.billing_address.city.evidence_ids = []
    draft.debtor.billing_address.city.ambiguity = "postal layout is unclear"

    class CombinedAddressOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[5] = OcrLine(
                text="10117 Berlin",
                confidence=0.99,
                bounding_box=result.lines[5].bounding_box,
            )
            return result

    outcome = ImageOrderExtractor(CombinedAddressOcr(), StubParser(draft)).extract(
        image_path
    )

    assert outcome.order is None
    assert any(
        issue.code == "AMBIGUOUS_FIELD"
        and issue.path == "debtor.billing_address.city"
        for issue in outcome.report.issues
    )


def test_conflicting_combined_and_separate_address_values_stop_authorization(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.billing_address.zip.value = "10117 Berlin"
    draft.debtor.billing_address.city.value = "Munich"

    class ConflictingAddressOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[5] = OcrLine(
                text="10117 Berlin",
                confidence=0.99,
                bounding_box=result.lines[5].bounding_box,
            )
            result.lines[6] = OcrLine(
                text="Munich",
                confidence=0.99,
                bounding_box=result.lines[6].bounding_box,
            )
            return result

    outcome = ImageOrderExtractor(ConflictingAddressOcr(), StubParser(draft)).extract(
        image_path
    )

    assert outcome.order is None
    assert any(
        issue.code == "CONFLICTING_ADDRESS_COMPONENTS"
        and issue.path == "debtor.billing_address"
        for issue in outcome.report.issues
    )


def test_partitioned_postal_city_normalizes_to_proven_format(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.billing_address.zip.value = "3011"
    draft.debtor.billing_address.zip.evidence_ids = ["ocr_0006"]
    draft.debtor.billing_address.city.value = "PZ Rotterdam"
    draft.debtor.billing_address.city.evidence_ids = ["ocr_0006"]
    draft.debtor.billing_address.country.value = "Netherlands"
    draft.debtor.billing_address.country.evidence_ids = ["ocr_0008"]

    class DutchAddressOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[5] = OcrLine(
                text="3011 PZ Rotterdam",
                confidence=0.99,
                bounding_box=result.lines[5].bounding_box,
            )
            result.lines[7] = OcrLine(
                text="Netherlands",
                confidence=0.99,
                bounding_box=result.lines[7].bounding_box,
            )
            return result

    outcome = ImageOrderExtractor(DutchAddressOcr(), StubParser(draft)).extract(
        image_path
    )

    assert outcome.order is not None
    assert outcome.order.debtor.billing_address.zip == "3011 PZ"
    assert outcome.order.debtor.billing_address.city == "Rotterdam"


def test_trusted_payment_claims_extraction(tmp_path: Path) -> None:
    from fakturama_automation.extraction.evidence import EvidenceDocument, EvidenceSpan
    from fakturama_automation.extraction.verification import trusted_payment_claims

    evidence = EvidenceDocument(
        document_id="doc1",
        source_path="order.png",
        source_sha256="abc",
        pages=[],
        spans=[
            EvidenceSpan(
                id="e1",
                text="PAYMENT METHOD",
                confidence=0.99,
                bbox=[100, 100, 300, 120],
                page=1,
                reading_order=1,
            ),
            EvidenceSpan(
                id="e2",
                text="Bank Transfer",
                confidence=0.99,
                bbox=[100, 130, 300, 150],
                page=1,
                reading_order=2,
            ),
            EvidenceSpan(
                id="e3",
                text="PAID STATUS",
                confidence=0.99,
                bbox=[400, 100, 600, 120],
                page=1,
                reading_order=3,
            ),
            EvidenceSpan(
                id="e4",
                text="PAID",
                confidence=0.99,
                bbox=[400, 130, 500, 150],
                page=1,
                reading_order=4,
            ),
            EvidenceSpan(
                id="e5",
                text="PAYMENT DATE",
                confidence=0.99,
                bbox=[700, 100, 900, 120],
                page=1,
                reading_order=5,
            ),
            EvidenceSpan(
                id="e6",
                text="2026-07-18",
                confidence=0.99,
                bbox=[700, 130, 850, 150],
                page=1,
                reading_order=6,
            ),
        ]
    )

    claims = trusted_payment_claims(evidence)
    by_path = {c.path: c.value for c in claims}
    assert by_path["payment.method"] == "Bank Transfer"
    assert by_path["payment.status"] == "PAID"
    assert by_path["payment.payment_date"] == "2026-07-18"


def test_unrecognized_country_does_not_enable_postal_format_guessing(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.billing_address.zip.value = "10117 Berlin"
    draft.debtor.billing_address.city.value = None
    draft.debtor.billing_address.city.evidence_ids = []
    draft.debtor.billing_address.country.value = "Unknownland"

    class UnknownCountryOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[5] = OcrLine(
                text="10117 Berlin",
                confidence=0.99,
                bounding_box=result.lines[5].bounding_box,
            )
            result.lines[7] = OcrLine(
                text="Unknownland",
                confidence=0.99,
                bounding_box=result.lines[7].bounding_box,
            )
            return result

    outcome = ImageOrderExtractor(UnknownCountryOcr(), StubParser(draft)).extract(
        image_path
    )

    assert outcome.order is None
    assert any(
        issue.code == "MISSING_REQUIRED_VALUE"
        and issue.path == "debtor.billing_address.city"
        for issue in outcome.report.issues
    )


def test_dates_currency_are_canonicalized_but_overdue_semantics_are_preserved(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.order_date.value = "14.08.2026"
    draft.currency.value = "eur"
    draft.payment.status.value = "OVERDUE"

    class CanonicalOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[0] = OcrLine(
                text="Order date: 14.08.2026",
                confidence=0.99,
                bounding_box=result.lines[0].bounding_box,
            )
            result.lines[9] = OcrLine(
                text="Status: OVERDUE",
                confidence=0.99,
                bounding_box=result.lines[9].bounding_box,
            )
            return result

    outcome = ImageOrderExtractor(CanonicalOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is not None
    assert outcome.order.order_date.isoformat() == "2026-08-14"
    assert outcome.order.currency == "EUR"
    assert outcome.order.payment.status.value == "OVERDUE"
    assert not outcome.report.valid
    assert any(
        issue.code == "UNSUPPORTED_PAYMENT_STATUS" for issue in outcome.report.issues
    )


def test_common_us_and_european_number_formats_share_one_parser() -> None:
    assert parse_decimal_text("USD 1,839.92") == Decimal("1839.92")
    assert parse_decimal_text("EUR 1.839,92") == Decimal("1839.92")
    assert parse_decimal_text("1 839,92") == Decimal("1839.92")
    assert decimal_values_in_text("NET TOTAL USD 1,839.92") == [Decimal("1839.92")]
    assert decimal_values_in_text("NET TOTAL EUR 1.839,92") == [Decimal("1839.92")]


def test_us_thousands_separators_are_grounded_and_canonicalized(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.items[0].unit_net_price.value = "500.00"
    draft.items[0].discount_percent.value = "0%"
    draft.items[0].source_total.value = "1,000.00"
    draft.totals.total_net.value = "1,000.00"
    draft.totals.total_vat.value = "190.00"
    draft.totals.total_gross.value = "1,190.00"

    replacements = {
        "unit": "500.00",
        "discount": "0%",
        "line_total": "1,000.00",
        "net": "Total net 1,000.00",
        "total_vat": "Total VAT 190.00",
        "gross": "Total gross 1,190.00 USD",
    }

    class ThousandsOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            return OcrResult(
                lines=[
                    OcrLine(
                        text=replacements.get(name, text),
                        confidence=0.96,
                        bounding_box=(10, index * 20, 200, index * 20 + 15),
                    )
                    for index, (name, text, _) in enumerate(OCR_VALUES, start=1)
                ]
            )

    outcome = ImageOrderExtractor(ThousandsOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is not None
    assert outcome.order.items[0].source_total == Decimal("1000.00")
    assert outcome.order.totals.total_net == Decimal("1000.00")
    assert outcome.order.totals.total_gross == Decimal("1190.00")
    assert outcome.report.valid
    assert not any(
        issue.code in {"UNGROUNDED_VALUE", "INVALID_EXTRACTED_VALUE"}
        for issue in outcome.report.issues
    )


def test_llm_json_parser_accepts_fences_and_short_prefaces() -> None:
    assert _parse_json_object('Result:\n```json\n{"status": "ok"}\n```') == {
        "status": "ok"
    }
    assert _parse_json_object('Here is the result: {"status": "ok"}') == {
        "status": "ok"
    }


def test_groq_strict_schema_requires_all_object_fields() -> None:
    schema = _strict_json_schema(CompactExtractionClaims.model_json_schema())

    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["FieldClaim"]["additionalProperties"] is False


def test_evidence_prompt_payload_is_compact_and_keeps_grounding_data(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    document = build_evidence_document(image_path, StubOcr().recognize(image_path))

    payload = document.prompt_payload()

    assert payload["span_columns"] == ["id", "text", "confidence", "bbox"]
    assert payload["spans"][0][0] == "ocr_0001"
    assert payload["spans"][0][1] == "Order date: 2026-08-14"
    assert "source_path" not in payload
    assert "source_sha256" not in payload
    assert "model" not in str(payload)


def test_item_table_is_inferred_from_headers_and_relative_geometry(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "table.png"
    Image.new("RGB", (1400, 600), "white").save(image_path)
    header_values = [
        ("#", (10, 100, 40, 130)),
        ("SKU", (90, 100, 170, 130)),
        ("DESCRIPTION", (280, 100, 520, 130)),
        ("QTY", (600, 100, 650, 130)),
        ("UNIT NET (USD)", (760, 100, 910, 130)),
        ("DISC.", (940, 100, 1000, 130)),
        ("VAT", (1040, 100, 1090, 130)),
        ("LINE NET (USD)", (1180, 100, 1360, 130)),
    ]
    row_values = [
        ("1", (10, 200, 40, 230)),
        ("MON-1", (90, 200, 170, 230)),
        ("Monitor", (280, 200, 520, 230)),
        ("4", (600, 200, 650, 230)),
        ("310.00", (760, 200, 910, 230)),
        ("5%", (940, 200, 1000, 230)),
        ("8%", (1040, 200, 1090, 230)),
        ("1,178.00", (1180, 200, 1360, 230)),
        ("2", (10, 300, 40, 330)),
        ("DOCK-1", (90, 300, 170, 330)),
        ("Dock", (280, 300, 520, 330)),
        ("4", (600, 300, 650, 330)),
        ("145.50", (760, 300, 910, 330)),
        ("0%", (940, 300, 1000, 330)),
        ("8%", (1040, 300, 1090, 330)),
        ("582.00", (1180, 300, 1360, 330)),
        ("NET TOTAL", (300, 450, 500, 480)),
    ]
    document = build_evidence_document(
        image_path,
        OcrResult(
            lines=[
                OcrLine(text, 0.99, bbox) for text, bbox in header_values + row_values
            ]
        ),
    )

    assert len(document.tables) == 1
    table = document.tables[0]
    assert [column.role for column in table.columns] == [
        "row_number",
        "sku",
        "description",
        "quantity",
        "unit_net_price",
        "discount_percent",
        "vat_percent",
        "source_total",
    ]
    assert len(table.rows) == 2
    second_cells = {cell.role: cell for cell in table.rows[1].cells}
    assert second_cells["row_number"].text == "2"
    assert second_cells["quantity"].text == "4"
    assert document.prompt_payload()["item_tables"][0]["rows"][1][0] == 1

    trusted_count, local_claims = trusted_item_claims(document)
    local_by_path = {claim.path: claim for claim in local_claims}
    assert trusted_count == 2
    assert local_by_path["items[0].source_total"].value == "1,178.00"
    assert local_by_path["items[1].quantity"].value == "4"
    assert local_by_path["items[1].description"].value == "Dock"
    assert "NET TOTAL" not in local_by_path["items[1].description"].value
    assert local_by_path["items[1].quantity"].evidence_ids == list(
        second_cells["quantity"].evidence_ids
    )

    report = ValidationReport()
    _validate_table_cell_claim(
        "items[1].quantity",
        list(second_cells["row_number"].evidence_ids),
        document.table_cell_index(),
        report,
    )
    assert [issue.code for issue in report.issues] == ["TABLE_CELL_MISMATCH"]

    correct_report = ValidationReport()
    _validate_table_cell_claim(
        "items[1].quantity",
        list(second_cells["quantity"].evidence_ids),
        document.table_cell_index(),
        correct_report,
    )
    assert correct_report.issues == []


def test_explicit_delivery_address_is_not_erased_when_claims_are_missing(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (500, 500), "white").save(image_path)
    evidence = build_evidence_document(image_path, StubOcr().recognize(image_path))
    billing_ids = ["ocr_0005", "ocr_0006", "ocr_0007", "ocr_0008"]
    duplicates = tuple(
        evidence.span_index()[evidence_id].model_copy(
            update={
                "id": f"duplicate_{evidence_id}",
                "bbox": tuple(
                    coordinate + 220 if index in {0, 2} else coordinate
                    for index, coordinate in enumerate(
                        evidence.span_index()[evidence_id].bbox
                    )
                ),
            }
        )
        for evidence_id in billing_ids
    )
    evidence = evidence.model_copy(update={"spans": evidence.spans + duplicates})
    draft = _valid_draft()
    draft.debtor.delivery_address = DraftAddress.model_validate(
        {"street": {}, "zip": {}, "city": {}, "country": {}}
    )

    order, report = ground_extraction_draft(draft, evidence)

    assert order is None
    assert {
        issue.path
        for issue in report.issues
        if issue.code == "MISSING_REQUIRED_VALUE"
    } == {
        "debtor.delivery_address.street",
        "debtor.delivery_address.zip",
        "debtor.delivery_address.city",
        "debtor.delivery_address.country",
    }


def test_wrapped_text_grounding_accepts_direct_span_continuation() -> None:
    spans = [
        EvidenceSpan(
            id="ocr_0001",
            text="buyer@schoenfeld-mueller-partner-",
            confidence=0.99,
            bbox=(0, 0, 300, 20),
            reading_order=0,
        ),
        EvidenceSpan(
            id="ocr_0002",
            text="fachgrosshandel.example.test",
            confidence=0.99,
            bbox=(0, 25, 300, 45),
            reading_order=1,
        ),
    ]

    assert _value_is_supported(
        "buyer@schoenfeld-mueller-partner-fachgrosshandel.example.test",
        spans,
        "text",
    )


def test_overdue_and_its_labeled_date_are_preserved_for_policy_review(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 520), "white").save(image_path)
    draft = _valid_draft()
    draft.payment.status.value = "OVERDUE"
    draft.payment.payment_date.value = "2025-12-05"
    draft.payment.payment_date.evidence_ids = ["ocr_0021"]

    class OverdueOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines[9] = OcrLine(
                text="OVERDUE",
                confidence=0.99,
                bounding_box=result.lines[9].bounding_box,
            )
            result.lines.append(
                OcrLine(
                    text="2025-12-05",
                    confidence=0.99,
                    bounding_box=(10, 440, 200, 455),
                )
            )
            return result

    outcome = ImageOrderExtractor(OverdueOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is not None
    assert outcome.order.payment.status.value == "OVERDUE"
    assert outcome.order.payment.payment_date.isoformat() == "2025-12-05"
    assert any(
        issue.code == "UNSUPPORTED_PAYMENT_STATUS" for issue in outcome.report.issues
    )


def test_compact_claims_expand_only_allowlisted_paths() -> None:
    response = CompactExtractionClaims.model_validate(
        {
            "item_count": 1,
            "delivery_address_present": False,
            "claims": [
                {
                    "path": "external_reference",
                    "value": "WEB-2026-0001",
                    "evidence_ids": ["ocr_0002"],
                    "ambiguity": None,
                },
                {
                    "path": "items[0].sku",
                    "value": "CH-1",
                    "evidence_ids": ["ocr_0011"],
                    "ambiguity": None,
                },
            ],
        }
    )

    draft = claims_to_draft(response)

    assert draft.external_reference.value == "WEB-2026-0001"
    assert draft.items[0].sku.value == "CH-1"
    assert draft.items[0].quantity.value is None


def test_compact_claims_reject_unknown_paths() -> None:
    response = CompactExtractionClaims.model_validate(
        {
            "item_count": 0,
            "delivery_address_present": False,
            "claims": [
                {
                    "path": "dangerous.untrusted_field",
                    "value": "invented",
                    "evidence_ids": ["ocr_0001"],
                    "ambiguity": None,
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="unsupported extraction claim path"):
        claims_to_draft(response)


def test_compact_claims_map_debtor_contact_aliases_to_billing_address() -> None:
    response = CompactExtractionClaims.model_validate(
        {
            "item_count": 0,
            "delivery_address_present": False,
            "claims": [
                {
                    "path": "debtor.email",
                    "value": "buyer@example.test",
                    "evidence_ids": ["ocr_0001"],
                    "ambiguity": None,
                },
                {
                    "path": "debtor.telephone",
                    "value": "+49 30 1234",
                    "evidence_ids": ["ocr_0002"],
                    "ambiguity": None,
                },
            ],
        }
    )

    draft = claims_to_draft(response)

    assert draft.debtor.billing_address.email.value == "buyer@example.test"
    assert draft.debtor.billing_address.telephone.value == "+49 30 1234"


def test_groq_strict_schema_is_not_duplicated_in_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    document = build_evidence_document(image_path, StubOcr().recognize(image_path))
    captured = {}

    class Response:
        ok = True

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "item_count": 0,
                                    "delivery_address_present": False,
                                    "claims": [],
                                }
                            )
                        }
                    }
                ]
            }

    def fake_post(endpoint, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(parser_module.requests, "post", fake_post)
    monkeypatch.setattr(parser_module, "verify_claims", lambda response, evidence: [])
    parser = CompatibleChatOrderParser(
        CompatibleChatConfig(
            base_url="https://api.groq.com/openai/v1",
            api_key="test-only",
            model="openai/gpt-oss-120b",
        )
    )

    parser.parse(document)

    assert captured["response_format"]["type"] == "json_schema"
    prompt = captured["messages"][-1]["content"]
    assert "JSON schema:" not in prompt
    assert "Preserve Unicode" in prompt
    assert "PARTIALLY PAID" in prompt
    assert "even if its values equal billing" in prompt
    assert "No trustworthy local item table was found" in prompt
    assert len(captured["messages"]) == 6
    first_example = json.loads(captured["messages"][2]["content"])
    assert first_example["delivery_address_present"] is True
    address_claims = {claim["path"]: claim for claim in first_example["claims"]}
    assert address_claims["debtor.billing_address.street"]["value"] == "Calle Mayor 8"
    assert address_claims["debtor.billing_address.zip"]["value"] == "28013"
    assert address_claims["debtor.billing_address.city"]["value"] == "Madrid"
    assert (
        address_claims["debtor.billing_address.zip"]["evidence_ids"]
        == address_claims["debtor.billing_address.city"]["evidence_ids"]
        == ["e4"]
    )
    assert address_claims["debtor.delivery_address.city"]["value"] == "Seville"
    second_example = json.loads(captured["messages"][4]["content"])
    payment_claims = {claim["path"]: claim for claim in second_example["claims"]}
    assert payment_claims["payment.status"]["value"] == "OVERDUE"
    assert payment_claims["payment.payment_date"]["value"] == "2026-05-01"


def test_schema_capability_is_explicit_and_conservative() -> None:
    groq = CompatibleChatConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key="test",
        model="openai/gpt-oss-120b",
    )
    unknown = CompatibleChatConfig(
        base_url="https://gateway.example/v1",
        api_key="test",
        model="strong-model",
    )
    forced = CompatibleChatConfig(
        base_url="https://gateway.example/v1",
        api_key="test",
        model="strong-model",
        supports_json_schema=True,
    )

    assert _supports_strict_json_schema(groq)
    assert not _supports_strict_json_schema(unknown)
    assert _supports_strict_json_schema(forced)


def test_parser_uses_local_table_cells_instead_of_llm_item_reconstruction(
    monkeypatch,
) -> None:
    cell_values = {
        "sku": "GEN-9",
        "description": "General item",
        "quantity": "2.5",
        "unit_net_price": "10.00",
        "discount_percent": "0%",
        "vat_percent": "19%",
        "source_total": "25.00",
    }
    spans = tuple(
        EvidenceSpan(
            id=f"e{index}",
            text=value,
            confidence=0.99,
            bbox=(index * 100, 100, index * 100 + 80, 120),
            reading_order=index,
        )
        for index, value in enumerate(cell_values.values(), start=1)
    )
    cells = tuple(
        EvidenceTableCell(role=role, text=value, evidence_ids=(f"e{index}",))
        for index, (role, value) in enumerate(cell_values.items(), start=1)
    )
    document = EvidenceDocument(
        document_id="test",
        source_path="test.png",
        source_sha256="0" * 64,
        pages=(EvidencePage(page=1, width=1000, height=500),),
        spans=spans,
        tables=(
            EvidenceTable(
                page=1,
                header_ids=(),
                columns=tuple(
                    EvidenceTableColumn(role=role, header_ids=(), x_range=(0, 1000))
                    for role in cell_values
                ),
                rows=(EvidenceTableRow(index=0, y_range=(90, 130), cells=cells),),
            ),
        ),
    )

    class Response:
        ok = True

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "item_count": 1,
                                    "delivery_address_present": False,
                                    "claims": [
                                        {
                                            "path": "items[0].quantity",
                                            "value": "999",
                                            "evidence_ids": ["e1"],
                                            "ambiguity": None,
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(parser_module.requests, "post", lambda *args, **kwargs: Response())
    parser = CompatibleChatOrderParser(
        CompatibleChatConfig(
            base_url="https://gateway.example/v1",
            api_key="test-only",
            model="text-model",
        )
    )

    draft = parser.parse(document)

    assert len(draft.items) == 1
    assert draft.items[0].quantity.value == "2.5"
    assert draft.items[0].quantity.evidence_ids == ["e3"]
    assert draft.items[0].sku.value == "GEN-9"


def test_parser_performs_one_focused_repair_and_preserves_unrelated_claims(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    document = build_evidence_document(image_path, StubOcr().recognize(image_path))
    responses = [
        {
            "item_count": 0,
            "delivery_address_present": False,
            "claims": [
                {
                    "path": "external_reference",
                    "value": "WRONG",
                    "evidence_ids": ["ocr_0001"],
                    "ambiguity": None,
                },
                {
                    "path": "currency",
                    "value": "EUR",
                    "evidence_ids": ["ocr_0003"],
                    "ambiguity": None,
                },
            ],
        },
        {
            "item_count": 99,
            "delivery_address_present": True,
            "claims": [
                {
                    "path": "external_reference",
                    "value": "WEB-2026-0001",
                    "evidence_ids": ["ocr_0002"],
                    "ambiguity": None,
                },
                {
                    "path": "currency",
                    "value": "USD",
                    "evidence_ids": ["ocr_0003"],
                    "ambiguity": None,
                },
            ],
        },
    ]
    calls = []

    class Response:
        ok = True

        def __init__(self, value):
            self.value = value

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(self.value)}}]}

    def fake_post(endpoint, **kwargs):
        calls.append(kwargs["json"])
        return Response(responses[len(calls) - 1])

    monkeypatch.setattr(parser_module.requests, "post", fake_post)
    parser = CompatibleChatOrderParser(
        CompatibleChatConfig(
            base_url="https://gateway.example/v1",
            api_key="test-only",
            model="text-model",
            supports_json_schema=False,
        )
    )

    draft = parser.parse(document)

    assert len(calls) == 2
    assert draft.external_reference.value == "WEB-2026-0001"
    assert draft.currency.value == "EUR"
    assert draft.debtor.delivery_address is None
    assert "Focused OCR evidence" in calls[1]["messages"][-1]["content"]


def test_strict_schema_rejection_falls_back_once_to_json_mode(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    document = build_evidence_document(image_path, StubOcr().recognize(image_path))
    calls = []

    class Rejected:
        ok = False
        status_code = 400

        @staticmethod
        def json():
            return {"error": {"message": "Invalid schema for response_format"}}

    class Accepted:
        ok = True

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"item_count":0,"delivery_address_present":false,"claims":[]}'
                        }
                    }
                ]
            }

    def fake_post(endpoint, **kwargs):
        calls.append(json.loads(json.dumps(kwargs["json"])))
        return Rejected() if len(calls) == 1 else Accepted()

    monkeypatch.setattr(parser_module.requests, "post", fake_post)
    monkeypatch.setattr(parser_module, "verify_claims", lambda response, evidence: [])
    parser = CompatibleChatOrderParser(
        CompatibleChatConfig(
            base_url="https://gateway.example/v1",
            api_key="test-only",
            model="text-model",
            supports_json_schema=True,
        )
    )

    parser.parse(document)
    parser._request_compact("Second request")

    assert len(calls) == 3
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"]["type"] == "json_object"
    assert "JSON schema:" in calls[1]["messages"][-1]["content"]
    assert calls[2]["response_format"]["type"] == "json_object"


def test_optional_placeholders_do_not_erase_required_values() -> None:
    response = CompactExtractionClaims.model_validate(
        {
            "item_count": 1,
            "delivery_address_present": False,
            "claims": [
                {"path": "debtor.alias", "value": "N/A", "evidence_ids": ["e1"], "ambiguity": None},
                {"path": "items[0].sku", "value": "-", "evidence_ids": ["e2"], "ambiguity": None},
            ],
        }
    )

    normalized = normalize_optional_placeholders(response)
    values = {claim.path: claim.value for claim in normalized.claims}
    assert values["debtor.alias"] is None
    assert values["items[0].sku"] == "-"


def test_exact_cited_payment_status_removes_only_unfounded_model_ambiguity(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "status.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    document = build_evidence_document(
        image_path,
        OcrResult(lines=[OcrLine("UNPAID", 0.99, (0, 0, 80, 20))]),
    )
    response = CompactExtractionClaims.model_validate(
        {
            "item_count": 0,
            "delivery_address_present": False,
            "claims": [
                {
                    "path": "payment.status",
                    "value": "UNPAID",
                    "evidence_ids": ["ocr_0001"],
                    "ambiguity": "status may be unclear",
                }
            ],
        }
    )

    normalized = normalize_proven_ambiguities(response, document)

    assert normalized.claims[0].ambiguity is None


def test_exact_labeled_payment_date_removes_descriptive_model_note(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "payment-date.png"
    Image.new("RGB", (300, 200), "white").save(image_path)
    document = build_evidence_document(
        image_path,
        OcrResult(
            lines=[
                OcrLine("PAYMENT DATE", 0.99, (100, 20, 240, 45)),
                OcrLine("2026-07-18", 0.99, (110, 55, 230, 80)),
            ]
        ),
    )
    response = CompactExtractionClaims.model_validate(
        {
            "item_count": 0,
            "delivery_address_present": False,
            "claims": [
                {
                    "path": "payment.payment_date",
                    "value": "2026-07-18",
                    "evidence_ids": ["ocr_0002"],
                    "ambiguity": "printed under PAYMENT DATE",
                }
            ],
        }
    )

    normalized = normalize_proven_ambiguities(response, document)

    assert normalized.claims[0].ambiguity is None


def test_unlabeled_payment_date_ambiguity_is_preserved(tmp_path: Path) -> None:
    image_path = tmp_path / "unlabeled-date.png"
    Image.new("RGB", (300, 200), "white").save(image_path)
    document = build_evidence_document(
        image_path,
        OcrResult(lines=[OcrLine("2026-07-18", 0.99, (110, 55, 230, 80))]),
    )
    response = CompactExtractionClaims.model_validate(
        {
            "item_count": 0,
            "delivery_address_present": False,
            "claims": [
                {
                    "path": "payment.payment_date",
                    "value": "2026-07-18",
                    "evidence_ids": ["ocr_0001"],
                    "ambiguity": "could be a due date",
                }
            ],
        }
    )

    normalized = normalize_proven_ambiguities(response, document)

    assert normalized.claims[0].ambiguity == "could be a due date"


def test_delivery_heading_with_explicit_no_shipment_marker_means_absent(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "delivery.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    document = build_evidence_document(
        image_path,
        OcrResult(
            lines=[
                OcrLine("Delivery Address", 0.99, (0, 0, 80, 20)),
                OcrLine("— (order refunded before shipment)", 0.99, (0, 25, 100, 45)),
            ]
        ),
    )
    response = CompactExtractionClaims.model_validate(
        {"item_count": 0, "delivery_address_present": True, "claims": []}
    )

    normalized = normalize_delivery_presence(response, document)

    assert not normalized.delivery_address_present


def test_delivery_presence_is_preserved_without_an_explicit_absence_marker(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "delivery.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    document = build_evidence_document(
        image_path,
        OcrResult(
            lines=[
                OcrLine("Delivery Address", 0.99, (0, 0, 80, 20)),
                OcrLine("Same as billing", 0.99, (0, 25, 100, 45)),
            ]
        ),
    )
    response = CompactExtractionClaims.model_validate(
        {"item_count": 0, "delivery_address_present": True, "claims": []}
    )

    normalized = normalize_delivery_presence(response, document)

    assert normalized.delivery_address_present


def test_clear_contact_name_can_be_split_with_shared_ocr_evidence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 600), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.first_name.value = "Marta"
    draft.debtor.first_name.evidence_ids = ["ocr_0021"]
    draft.debtor.last_name.value = "Klein"
    draft.debtor.last_name.evidence_ids = ["ocr_0021"]

    class ContactNameOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines.append(
                OcrLine(
                    text="Marta Klein",
                    confidence=0.99,
                    bounding_box=(10, 500, 200, 515),
                )
            )
            return result

    outcome = ImageOrderExtractor(ContactNameOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is not None
    assert outcome.order.debtor.first_name == "Marta"
    assert outcome.order.debtor.last_name == "Klein"
    assert outcome.order.evidence["debtor.first_name"].evidence_ids == ["ocr_0021"]
    assert outcome.order.evidence["debtor.last_name"].evidence_ids == ["ocr_0021"]


def test_overlapping_shared_span_name_split_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 600), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.first_name.value = "Marta"
    draft.debtor.first_name.evidence_ids = ["ocr_0021"]
    draft.debtor.last_name.value = "arta"
    draft.debtor.last_name.evidence_ids = ["ocr_0021"]

    class ContactNameOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines.append(
                OcrLine(
                    text="Marta Klein",
                    confidence=0.99,
                    bounding_box=(10, 500, 200, 515),
                )
            )
            return result

    outcome = ImageOrderExtractor(ContactNameOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is None
    assert any(
        issue.code == "INVALID_PERSON_NAME_SPLIT" for issue in outcome.report.issues
    )


def test_shared_name_label_with_separate_value_spans_is_allowed(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 650), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.first_name.value = "Marta"
    draft.debtor.first_name.evidence_ids = ["ocr_0021", "ocr_0022"]
    draft.debtor.last_name.value = "Klein"
    draft.debtor.last_name.evidence_ids = ["ocr_0021", "ocr_0023"]

    class SeparateNameSpansOcr(StubOcr):
        def recognize(self, image_path: Path) -> OcrResult:
            result = super().recognize(image_path)
            result.lines.extend(
                [
                    OcrLine("CONTACT NAME", 0.99, (10, 500, 200, 515)),
                    OcrLine("Marta", 0.99, (10, 520, 200, 535)),
                    OcrLine("Klein", 0.99, (10, 540, 200, 555)),
                ]
            )
            return result

    outcome = ImageOrderExtractor(SeparateNameSpansOcr(), StubParser(draft)).extract(
        image_path
    )

    assert outcome.order is not None
    assert outcome.order.debtor.first_name == "Marta"
    assert outcome.order.debtor.last_name == "Klein"


def test_genuinely_ambiguous_contact_name_still_requires_review(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.company.value = None
    draft.debtor.company.evidence_ids = []
    draft.debtor.first_name.ambiguity = "Personal name order cannot be determined"
    draft.debtor.last_name.ambiguity = "Personal name order cannot be determined"

    outcome = ImageOrderExtractor(StubOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is None
    assert any(
        issue.code == "MISSING_DEBTOR_IDENTITY" for issue in outcome.report.issues
    )


def test_optional_ambiguous_contact_does_not_block_a_supported_company(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (300, 500), "white").save(image_path)
    draft = _valid_draft()
    draft.debtor.first_name.value = None
    draft.debtor.first_name.ambiguity = "multi-part name was not split"
    draft.debtor.last_name.value = None
    draft.debtor.last_name.ambiguity = "multi-part name was not split"

    outcome = ImageOrderExtractor(StubOcr(), StubParser(draft)).extract(image_path)

    assert outcome.order is not None
    assert outcome.order.debtor.company == draft.debtor.company.value
    assert not any(
        issue.path in {"debtor.first_name", "debtor.last_name"}
        for issue in outcome.report.issues
    )


def test_numeric_evidence_requires_an_exact_ocr_line() -> None:
    assert _evidence_text_matches("2", "2")
    assert not _evidence_text_matches("2", "2026 07 18")
    assert not _evidence_text_matches("2", "eur 108 30")
    assert not _evidence_text_matches("web 2026 0714 a17", "2")
    assert not _evidence_text_matches("eur 570 00", "eur")
    assert _evidence_text_matches("10117", "10117 berlin")


def test_preprocessor_flags_low_resolution_document(tmp_path: Path) -> None:
    image_path = tmp_path / "small.png"
    Image.new("RGB", (480, 640), "white").save(image_path)

    prepared = OpenCvPreprocessor(minimum_width=960).prepare(image_path)

    assert prepared.scale_x == 2
    assert any("low resolution" in issue for issue in prepared.quality.issues)


def test_review_packet_contains_uncertain_field_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "order.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    order = OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )
    order = order.model_copy(
        update={
            "evidence": {
                "items[0].sku": FieldEvidence(
                    source_text=order.items[0].sku,
                    confidence=0.80,
                    bounding_box=(20, 20, 80, 40),
                )
            }
        }
    )
    report = ValidationReport(
        issues=[
            ValidationIssue(
                path="items[0].sku",
                message="review identifier",
                severity=Severity.WARNING,
            )
        ]
    )

    packet_path = write_review_packet(
        image_path, order, report, tmp_path / "review.json"
    )
    packet = packet_path.read_text(encoding="utf-8")

    assert "HUMAN_REVIEW_REQUIRED" in packet
    assert "source_crop" in packet
    assert list((tmp_path / "review-assets").glob("*.png"))
