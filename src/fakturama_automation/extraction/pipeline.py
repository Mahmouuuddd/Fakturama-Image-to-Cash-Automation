from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Literal

from pydantic import ValidationError

from fakturama_automation.domain.matching import normalize_text
from fakturama_automation.domain.models import FieldEvidence, OrderInput
from fakturama_automation.domain.numbers import (
    canonical_decimal,
    decimal_values_in_text,
    parse_decimal_text,
)
from fakturama_automation.domain.validation import (
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_order,
)

from .evidence import (
    DraftAddress,
    EvidenceDocument,
    EvidenceSpan,
    ExtractedField,
    ExtractionDraft,
    build_evidence_document,
)
from .ocr import OcrBackend, OcrResult
from .parser import StructuredOrderParser


@dataclass(frozen=True)
class ExtractionOutcome:
    """Complete result of local OCR, semantic extraction, and local grounding."""

    evidence: EvidenceDocument
    draft: ExtractionDraft
    order: OrderInput | None
    report: ValidationReport
    ocr: OcrResult

    def __iter__(self):
        """Keep the former tuple-unpacking shape during the migration."""
        yield self.order
        yield self.report
        yield self.ocr


class JsonOrderExtractor:
    def extract(self, path: Path) -> tuple[OrderInput, ValidationReport]:
        data = json.loads(path.read_text(encoding="utf-8"))
        order = OrderInput.model_validate(data)
        return order, validate_order(order)


class ImageOrderExtractor:
    """Trusted image path: local OCR -> evidence -> draft -> local grounding."""

    def __init__(self, ocr: OcrBackend, parser: StructuredOrderParser) -> None:
        self.ocr = ocr
        self.parser = parser

    def extract(
        self,
        image_path: Path,
        *,
        evidence_callback: Callable[[EvidenceDocument], None] | None = None,
    ) -> ExtractionOutcome:
        ocr_result = self.ocr.recognize(image_path)
        evidence = build_evidence_document(image_path, ocr_result)
        if evidence_callback is not None:
            evidence_callback(evidence)
        draft = self.parser.parse(evidence)
        order, report = ground_extraction_draft(draft, evidence)

        if order is not None:
            report.issues.extend(validate_order(order, require_evidence=True).issues)

        for warning in evidence.preprocessing_warnings:
            report.issues.append(
                ValidationIssue(
                    code="IMAGE_QUALITY_WARNING",
                    path="image_quality",
                    message=warning,
                    severity=Severity.WARNING,
                )
            )
            if warning.startswith("low resolution"):
                for index, _ in enumerate(draft.items):
                    report.issues.append(
                        ValidationIssue(
                            code="LOW_RESOLUTION_IDENTIFIER_REVIEW",
                            path=f"items[{index}].sku",
                            message=(
                                "critical identifier requires visual confirmation "
                                "because the source image is low resolution"
                            ),
                            severity=Severity.WARNING,
                        )
                    )

        return ExtractionOutcome(
            evidence=evidence,
            draft=draft,
            order=order,
            report=report,
            ocr=ocr_result,
        )


FieldKind = Literal["text", "number", "percentage", "date", "status", "currency"]


def ground_extraction_draft(
    draft: ExtractionDraft, evidence: EvidenceDocument
) -> tuple[OrderInput | None, ValidationReport]:
    """Convert a nullable LLM draft into business input only when locally grounded."""
    report = ValidationReport()
    index = evidence.span_index()
    table_cell_index = evidence.table_cell_index()
    grounded: dict[str, FieldEvidence] = {}

    def value(
        field: ExtractedField,
        path: str,
        *,
        required: bool = False,
        kind: FieldKind = "text",
        default: str | None = None,
    ) -> str | None:
        if field.ambiguity and (required or field.value is not None):
            report.issues.append(
                ValidationIssue(
                    code="AMBIGUOUS_FIELD",
                    path=path,
                    message=field.ambiguity,
                    severity=Severity.WARNING if field.value is not None else Severity.ERROR,
                )
            )
        if field.value is None or not str(field.value).strip():
            if required and not field.ambiguity:
                report.issues.append(
                    ValidationIssue(
                        code="MISSING_REQUIRED_VALUE",
                        path=path,
                        message="required value was not extracted",
                    )
                )
            return default

        extracted = str(field.value).strip()
        grounded_value = _canonical_grounded_value(extracted, kind)
        if not field.evidence_ids:
            report.issues.append(
                ValidationIssue(
                    code="MISSING_EVIDENCE",
                    path=path,
                    message="extracted value has no OCR evidence IDs",
                )
            )
            return grounded_value

        unknown_ids = [item for item in field.evidence_ids if item not in index]
        if unknown_ids:
            report.issues.append(
                ValidationIssue(
                    code="UNKNOWN_EVIDENCE_ID",
                    path=path,
                    message="unknown OCR evidence IDs: " + ", ".join(unknown_ids),
                )
            )
            return grounded_value

        spans = [index[item] for item in field.evidence_ids]
        _validate_table_cell_claim(path, field.evidence_ids, table_cell_index, report)
        if not _value_is_supported(grounded_value, spans, kind):
            report.issues.append(
                ValidationIssue(
                    code="UNGROUNDED_VALUE",
                    path=path,
                    message=(
                        f"extracted value {extracted!r} is not supported by the referenced "
                        "OCR text"
                    ),
                )
            )
            return grounded_value

        grounded[path] = _field_evidence(spans, field.evidence_ids)
        return grounded_value

    def address(item: DraftAddress, prefix: str) -> dict:
        item = _normalize_address_claims(item, index, report, prefix)
        return {
            "street": value(item.street, f"{prefix}.street", required=True),
            "zip": value(item.zip, f"{prefix}.zip", required=True),
            "city": value(item.city, f"{prefix}.city", required=True),
            "country": value(item.country, f"{prefix}.country", required=True),
            "email": value(item.email, f"{prefix}.email"),
            "telephone": value(item.telephone, f"{prefix}.telephone"),
            "additional_name": value(item.additional_name, f"{prefix}.additional_name"),
            "address_specification": value(
                item.address_specification, f"{prefix}.address_specification"
            ),
            "district": value(item.district, f"{prefix}.district"),
        }

    billing = address(draft.debtor.billing_address, "debtor.billing_address")
    delivery_draft = draft.debtor.delivery_address
    delivery = (
        address(delivery_draft, "debtor.delivery_address")
        if delivery_draft
        else None
    )
    items: list[dict] = []
    if not draft.items:
        report.issues.append(
            ValidationIssue(
                code="MISSING_ORDER_ITEMS",
                path="items",
                message="no order items were extracted",
            )
        )
    for item_index, item in enumerate(draft.items):
        prefix = f"items[{item_index}]"
        items.append(
            {
                "sku": value(item.sku, f"{prefix}.sku", required=True),
                "description": value(
                    item.description, f"{prefix}.description", required=True
                ),
                "quantity": value(
                    item.quantity, f"{prefix}.quantity", required=True, kind="number"
                ),
                "unit_net_price": value(
                    item.unit_net_price,
                    f"{prefix}.unit_net_price",
                    required=True,
                    kind="number",
                ),
                "vat_percent": value(
                    item.vat_percent,
                    f"{prefix}.vat_percent",
                    required=True,
                    kind="percentage",
                ),
                "discount_percent": value(
                    item.discount_percent,
                    f"{prefix}.discount_percent",
                    required=True,
                    kind="percentage",
                ),
                "source_total": value(
                    item.source_total,
                    f"{prefix}.source_total",
                    required=True,
                    kind="number",
                ),
            }
        )

    debtor_company = value(draft.debtor.company, "debtor.company", default="")
    debtor_first_name = value(
        draft.debtor.first_name, "debtor.first_name", default=""
    )
    debtor_last_name = value(draft.debtor.last_name, "debtor.last_name", default="")
    if not debtor_company and not debtor_first_name and not debtor_last_name:
        report.issues.append(
            ValidationIssue(
                code="MISSING_DEBTOR_IDENTITY",
                path="debtor",
                message="debtor requires a company or supported contact name",
            )
        )
    _validate_shared_name_split(
        draft.debtor.first_name,
        draft.debtor.last_name,
        index,
        report,
    )
    payment_status = value(
        draft.payment.status, "payment.status", required=True, kind="status"
    )
    payment_date = value(draft.payment.payment_date, "payment.payment_date", kind="date")

    raw_order = {
        "order_date": value(draft.order_date, "order_date", required=True, kind="date"),
        "external_reference": value(
            draft.external_reference, "external_reference", required=True
        ),
        "currency": value(draft.currency, "currency", required=True, kind="currency"),
        "debtor": {
            "company": debtor_company,
            "first_name": debtor_first_name,
            "last_name": debtor_last_name,
            "alias": value(draft.debtor.alias, "debtor.alias", default=""),
            "salutation": value(draft.debtor.salutation, "debtor.salutation"),
            "billing_address": billing,
            "delivery_address": delivery,
        },
        "payment": {
            "method": value(draft.payment.method, "payment.method", required=True),
            "status": payment_status,
            "payment_date": payment_date,
        },
        "items": items,
        "totals": {
            "total_net": value(
                draft.totals.total_net, "totals.total_net", required=True, kind="number"
            ),
            "total_vat": value(
                draft.totals.total_vat, "totals.total_vat", required=True, kind="number"
            ),
            "total_gross": value(
                draft.totals.total_gross,
                "totals.total_gross",
                required=True,
                kind="number",
            ),
        },
        "evidence": grounded,
    }

    # Grounding and ambiguity issues are already causal and field-specific.
    # Do not add downstream Pydantic conversion noise for an unauthorized draft.
    if not report.valid:
        return None, report

    try:
        candidate = OrderInput.model_validate(raw_order)
    except ValidationError as exc:
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            report.issues.append(
                ValidationIssue(
                    code="INVALID_EXTRACTED_VALUE",
                    path=_pydantic_error_path(error.get("loc", ())),
                    message=str(error.get("msg", "invalid extracted value")),
                )
            )
        candidate = None
    except (InvalidOperation, ValueError, TypeError) as exc:
        report.issues.append(
            ValidationIssue(
                code="INVALID_EXTRACTED_VALUE",
                path="extraction",
                message=str(exc),
            )
        )
        candidate = None

    if not report.valid:
        return None, report
    return candidate, report


def _canonical_grounded_value(value: str, kind: FieldKind) -> str:
    """Apply only lossless or explicitly approved business normalizations.

    Grounding still checks the canonical result against the cited OCR spans, so
    normalization cannot turn unsupported model output into authorized data.
    """
    if kind in {"number", "percentage"}:
        return canonical_decimal(value) or value
    if kind == "date":
        parsed = _date_value(value)
        return parsed.isoformat() if parsed is not None else value
    if kind == "currency":
        return value.strip().upper()
    if kind == "status":
        return _normalize_source_text(value).upper()
    return value


_PREFIX_POSTAL_PATTERNS = (
    (
        {"de", "deutschland", "germany"},
        re.compile(r"^(?P<postal>\d{5})\s+(?P<city>.+)$", re.IGNORECASE),
    ),
    (
        {
            "at",
            "austria",
            "be",
            "belgium",
            "belgique",
            "belgie",
            "belgië",
            "ch",
            "danmark",
            "denmark",
            "hu",
            "hungary",
            "no",
            "norge",
            "norway",
            "österreich",
            "schweiz",
            "suisse",
            "svizzera",
            "switzerland",
        },
        re.compile(r"^(?P<postal>\d{4})\s+(?P<city>.+)$", re.IGNORECASE),
    ),
    (
        {
            "es",
            "espana",
            "españa",
            "fi",
            "finland",
            "fr",
            "france",
            "it",
            "italia",
            "italy",
            "se",
            "spain",
            "sweden",
            "sverige",
        },
        re.compile(r"^(?P<postal>\d{3}\s?\d{2})\s+(?P<city>.+)$", re.IGNORECASE),
    ),
    (
        {"nl", "netherlands", "nederland"},
        re.compile(
            r"^(?P<postal>\d{4}\s?[A-Z]{2})\s+(?P<city>.+)$", re.IGNORECASE
        ),
    ),
    (
        {"pl", "poland", "polska"},
        re.compile(r"^(?P<postal>\d{2}-\d{3})\s+(?P<city>.+)$", re.IGNORECASE),
    ),
    (
        {
            "cz",
            "czech republic",
            "czechia",
            "sk",
            "slovakia",
            "slovensko",
        },
        re.compile(r"^(?P<postal>\d{3}\s?\d{2})\s+(?P<city>.+)$", re.IGNORECASE),
    ),
    (
        {"jp", "japan", "日本"},
        re.compile(r"^(?P<postal>\d{3}-\d{4})\s+(?P<city>.+)$", re.IGNORECASE),
    ),
)
_SUFFIX_POSTAL_PATTERNS = (
    (
        {"us", "usa", "united states", "united states of america"},
        re.compile(
            r"^(?P<city>.+?)[,\s]+(?P<postal>\d{5}(?:-\d{4})?)$",
            re.IGNORECASE,
        ),
    ),
    (
        {"ca", "canada"},
        re.compile(
            r"^(?P<city>.+?)[,\s]+(?P<postal>[A-Z]\d[A-Z]\s?\d[A-Z]\d)$",
            re.IGNORECASE,
        ),
    ),
    (
        {"england", "gb", "great britain", "uk", "united kingdom"},
        re.compile(
            r"^(?P<city>.+?)[,\s]+(?P<postal>[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})$",
            re.IGNORECASE,
        ),
    ),
    (
        {"au", "australia", "new zealand", "nz"},
        re.compile(r"^(?P<city>.+?)[,\s]+(?P<postal>\d{4})$", re.IGNORECASE),
    ),
    (
        {"in", "india"},
        re.compile(r"^(?P<city>.+?)[,\s]+(?P<postal>\d{6})$", re.IGNORECASE),
    ),
    (
        {"br", "brazil"},
        re.compile(r"^(?P<city>.+?)[,\s]+(?P<postal>\d{5}-\d{3})$", re.IGNORECASE),
    ),
)


def _normalize_address_claims(
    address: DraftAddress,
    evidence_index: dict[str, EvidenceSpan],
    report: ValidationReport,
    prefix: str,
) -> DraftAddress:
    """Split a proven combined postal/city claim without guessing an address.

    A repair is allowed only when one of the two fields already has a grounded
    claim and its own cited span contains a single country-compatible postal/city
    pair. Conflicting candidates and explicit model ambiguity remain untouched.
    """
    if address.zip.ambiguity or address.city.ambiguity:
        return address
    country = str(address.country.value or "").strip()
    if not country:
        return address

    candidates: list[tuple[str, str, list[str]]] = []
    for field in (address.zip, address.city):
        if field.value is not None and str(field.value).strip():
            parsed = _split_postal_city(str(field.value), country)
            if parsed is not None:
                candidates.append((*parsed, list(field.evidence_ids)))
        for evidence_id in field.evidence_ids:
            span = evidence_index.get(evidence_id)
            if span is None:
                continue
            parsed = _split_postal_city(span.text, country)
            if parsed is not None:
                candidates.append((*parsed, list(field.evidence_ids)))

    unique: dict[tuple[str, str], list[str]] = {}
    for postal, city, evidence_ids in candidates:
        unique.setdefault((postal, city), evidence_ids)
    if len(unique) > 1:
        report.issues.append(
            ValidationIssue(
                code="CONFLICTING_ADDRESS_COMPONENTS",
                path=prefix,
                message="cited postal/city claims contain conflicting combinations",
            )
        )
        return address
    if not unique:
        return address
    (postal, city), candidate_ids = next(iter(unique.items()))
    if not candidate_ids:
        return address

    current_postal = str(address.zip.value or "").strip()
    current_city = str(address.city.value or "").strip()
    postal_matches = not current_postal or _component_matches(
        current_postal, postal, city
    )
    city_matches = not current_city or _component_matches(current_city, city, postal)
    if not postal_matches or not city_matches:
        report.issues.append(
            ValidationIssue(
                code="CONFLICTING_ADDRESS_COMPONENTS",
                path=prefix,
                message=(
                    "separate postal/city claims conflict with a combined value in "
                    "their cited OCR evidence"
                ),
            )
        )
        return address
    if not current_postal and not current_city:
        return address

    postal_ids = address.zip.evidence_ids or candidate_ids
    city_ids = address.city.evidence_ids or candidate_ids
    return address.model_copy(
        update={
            "zip": address.zip.model_copy(
                update={"value": postal, "evidence_ids": list(postal_ids)}
            ),
            "city": address.city.model_copy(
                update={"value": city, "evidence_ids": list(city_ids)}
            ),
        }
    )


def _component_matches(value: str, component: str, other: str) -> bool:
    normalized = normalize_text(value).strip(" ,")
    expected = normalize_text(component).strip(" ,")
    combined = {
        normalize_text(f"{component} {other}").strip(" ,"),
        normalize_text(f"{other} {component}").strip(" ,"),
        normalize_text(f"{component}, {other}").strip(" ,"),
        normalize_text(f"{other}, {component}").strip(" ,"),
    }
    return normalized == expected or normalized in combined


def _split_postal_city(value: str, country: str) -> tuple[str, str] | None:
    source = _normalize_source_text(value).strip(" ,")
    country_key = normalize_text(country)
    pattern = next(
        (
            candidate
            for countries, candidate in (
                *_PREFIX_POSTAL_PATTERNS,
                *_SUFFIX_POSTAL_PATTERNS,
            )
            if country_key in countries
        ),
        None,
    )
    if pattern is None or (match := pattern.fullmatch(source)) is None:
        return None

    postal = _normalize_source_text(match.group("postal")).upper()
    city = _normalize_source_text(match.group("city")).strip(" ,")
    if not any(character.isdigit() for character in postal):
        return None
    if not any(character.isalpha() for character in city):
        return None
    return postal, city


def _required_address_is_empty(address: DraftAddress) -> bool:
    return all(
        field.value is None or not str(field.value).strip()
        for field in (address.street, address.zip, address.city, address.country)
    )


def _address_values_repeat_elsewhere(
    address: DraftAddress, evidence: EvidenceDocument
) -> bool:
    required = (address.street, address.zip, address.city, address.country)
    if any(field.value is None or not str(field.value).strip() for field in required):
        return False
    source_ids = {
        evidence_id for field in required for evidence_id in field.evidence_ids
    }
    for field in required:
        extracted = str(field.value).strip()
        if not any(
            span.id not in source_ids
            and _value_is_supported(extracted, [span], "text")
            for span in evidence.spans
        ):
            return False
    return True


def _validate_shared_name_split(
    first_name: ExtractedField,
    last_name: ExtractedField,
    evidence_index: dict[str, EvidenceSpan],
    report: ValidationReport,
) -> None:
    """Reject overlapping name parts when the LLM split one OCR value span.

    The LLM retains responsibility for the semantic split. This local check only
    verifies that two claims sharing evidence point to distinct source substrings.
    """
    if not first_name.value or not last_name.value:
        return
    shared_ids = set(first_name.evidence_ids) & set(last_name.evidence_ids)
    shared_spans = []
    seen_ids: set[str] = set()
    for evidence_id in first_name.evidence_ids:
        if (
            evidence_id in shared_ids
            and evidence_id in evidence_index
            and evidence_id not in seen_ids
        ):
            shared_spans.append(evidence_index[evidence_id])
            seen_ids.add(evidence_id)
    if not shared_spans:
        return

    source = _normalize_source_text(" ".join(span.text for span in shared_spans))
    first = _normalize_source_text(str(first_name.value))
    last = _normalize_source_text(str(last_name.value))
    if first not in source or last not in source:
        # The common evidence can be a label while separate cited spans support
        # each value. Individual field grounding has already validated that case.
        return
    if _has_disjoint_occurrences(source, first, last):
        return

    report.issues.append(
        ValidationIssue(
            code="INVALID_PERSON_NAME_SPLIT",
            path="debtor.first_name",
            message=(
                "first and last name claims sharing OCR evidence must be distinct, "
                "non-overlapping source substrings"
            ),
        )
    )


_ITEM_TABLE_ROLES = {
    "sku": "sku",
    "description": "description",
    "quantity": "quantity",
    "unit_net_price": "unit_net_price",
    "vat_percent": "vat_percent",
    "discount_percent": "discount_percent",
    "source_total": "source_total",
}


def _validate_table_cell_claim(
    path: str,
    evidence_ids: list[str],
    cell_index: dict[str, tuple[int, str]],
    report: ValidationReport,
) -> None:
    match = re.fullmatch(r"items\[(\d+)]\.([a-z_]+)", path)
    if match is None or not cell_index:
        return
    item_index, field = int(match.group(1)), match.group(2)
    expected_role = _ITEM_TABLE_ROLES.get(field)
    if expected_role is None:
        return
    mapped = [
        (evidence_id, *cell_index[evidence_id])
        for evidence_id in evidence_ids
        if evidence_id in cell_index
    ]
    if not mapped:
        return
    mismatched = [
        (evidence_id, row, role)
        for evidence_id, row, role in mapped
        if row != item_index or role != expected_role
    ]
    if not mismatched:
        return
    observed = ", ".join(
        f"{evidence_id}=item[{row}].{role}"
        for evidence_id, row, role in mismatched
    )
    report.issues.append(
        ValidationIssue(
            code="TABLE_CELL_MISMATCH",
            path=path,
            message=(
                f"expected evidence from item[{item_index}].{expected_role}; "
                f"received {observed}"
            ),
        )
    )


def _has_disjoint_occurrences(source: str, first: str, last: str) -> bool:
    if not source or not first or not last:
        return False
    first_ranges = [match.span() for match in re.finditer(re.escape(first), source)]
    last_ranges = [match.span() for match in re.finditer(re.escape(last), source)]
    return any(
        first_end <= last_start or last_end <= first_start
        for first_start, first_end in first_ranges
        for last_start, last_end in last_ranges
    )


def _field_evidence(spans: list[EvidenceSpan], evidence_ids: list[str]) -> FieldEvidence:
    pages = {span.page for span in spans}
    box = None
    if len(pages) == 1:
        box = (
            min(span.bbox[0] for span in spans),
            min(span.bbox[1] for span in spans),
            max(span.bbox[2] for span in spans),
            max(span.bbox[3] for span in spans),
        )
    return FieldEvidence(
        source_text=" | ".join(span.text for span in spans),
        confidence=min(span.confidence for span in spans),
        bounding_box=box,
        evidence_ids=evidence_ids,
        page=spans[0].page,
    )


def _value_is_supported(value: str, spans: list[EvidenceSpan], kind: FieldKind) -> bool:
    source = " ".join(span.text for span in spans)
    if kind in {"number", "percentage"}:
        expected = _decimal_value(value)
        if expected is None:
            return False
        return any(number == expected for number in _numbers_in_text(source))
    if kind == "date":
        expected_date = _date_value(value)
        return expected_date is not None and expected_date in _dates_in_text(source)
    if kind == "currency":
        expected_currency = value.strip().upper()
        currency_aliases = {
            "EUR": ("eur", "€", "euro"),
            "USD": ("usd", "$", "dollar"),
            "GBP": ("gbp", "£", "pound"),
        }
        normalized_source = unicodedata.normalize("NFKC", source).casefold()
        aliases = currency_aliases.get(expected_currency, (expected_currency.casefold(),))
        return any(alias in normalized_source for alias in aliases)
    if kind == "status":
        normalized_source = _normalize_source_text(source).casefold()
        expected_status = value.strip().upper()
        supported = {
            "PAID": r"\bpaid\b",
            "UNPAID": r"\bunpaid\b",
            "OVERDUE": r"\boverdue\b",
            "PARTIALLY PAID": r"\bpartially\s+paid\b",
            "REFUNDED": r"\brefunded\b",
        }
        pattern = supported.get(expected_status)
        if pattern is None:
            return False
        if expected_status == "PAID" and re.search(
            r"\b(?:unpaid|partially\s+paid)\b", normalized_source
        ):
            return False
        return bool(re.search(pattern, normalized_source))
    normalized_value = _normalize_source_text(value)
    source_variants = {
        _normalize_source_text(source),
        _normalize_source_text("".join(span.text for span in spans)),
    }
    return bool(normalized_value) and any(
        normalized_value in candidate for candidate in source_variants
    )


def _decimal_value(value: str) -> Decimal | None:
    return parse_decimal_text(value)


def _numbers_in_text(text: str) -> list[Decimal]:
    return decimal_values_in_text(text)


def _date_value(value: str) -> date | None:
    for format_string in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), format_string).date()
        except ValueError:
            continue
    return None


def _dates_in_text(text: str) -> set[date]:
    candidates = re.findall(r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b", text)
    return {parsed for candidate in candidates if (parsed := _date_value(candidate)) is not None}


def _pydantic_error_path(location: tuple) -> str:
    path = ""
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += ("." if path else "") + str(part)
    return path or "extraction"


def _normalize_source_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    return re.sub(r"\s+", " ", normalized)


def _evidence_text_matches(expected: str, observed: str) -> bool:
    """Legacy matching helper retained for diagnostic-parser compatibility tests."""
    if not observed:
        return False
    if expected.isdigit() and len(expected) >= 4:
        return bool(re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", observed))
    if (
        len(expected) < 4
        or len(observed) < 4
        or expected.replace(".", "").isdigit()
        or observed.replace(".", "").isdigit()
    ):
        return expected == observed
    return expected in observed or observed in expected
