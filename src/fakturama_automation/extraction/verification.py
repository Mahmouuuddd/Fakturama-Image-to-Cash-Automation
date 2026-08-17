from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from fakturama_automation.domain.numbers import decimal_values_in_text, parse_decimal_text

from .evidence import CompactExtractionClaims, EvidenceDocument, EvidenceSpan, FieldClaim


_ITEM_ROLES = {
    "sku",
    "description",
    "quantity",
    "unit_net_price",
    "vat_percent",
    "discount_percent",
    "source_total",
}
_NUMERIC_ITEM_ROLES = {
    "quantity",
    "unit_net_price",
    "vat_percent",
    "discount_percent",
    "source_total",
}
_OPTIONAL_PLACEHOLDER_PATHS = {
    "debtor.company",
    "debtor.first_name",
    "debtor.last_name",
    "debtor.alias",
    "debtor.salutation",
    "payment.payment_date",
}
_PLACEHOLDERS = {"-", "--", "—", "n/a", "na", "not applicable"}
_ROOT_PATHS = {
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
_REQUIRED_PATHS = {
    "order_date",
    "external_reference",
    "currency",
    "debtor.billing_address.street",
    "debtor.billing_address.zip",
    "debtor.billing_address.city",
    "debtor.billing_address.country",
    "payment.method",
    "payment.status",
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
_TOTAL_LABELS = {
    "totals.total_net": {
        "net total",
        "total net",
        "subtotal",
        "summe netto",
    },
    "totals.total_vat": {
        "vat total",
        "total vat",
        "tax total",
        "mwst gesamt",
        "ust gesamt",
    },
    "totals.total_gross": {
        "gross total",
        "total gross",
        "grand total",
        "gesamt",
    },
}
_BILLING_ADDRESS_LABELS = {"billing address", "bill to", "invoice address"}
_DELIVERY_ADDRESS_LABELS = {"delivery address", "shipping address", "ship to"}
_COMPANY_LABELS = {"company", "customer company", "business name"}
_CONTACT_NAME_LABELS = {
    "contact name",
    "contact person",
    "contact",
    "customer name",
    "name",
    "ansprechpartner",
}
_ADDRESS_SECTION_BOUNDARIES = {
    "items",
    "order items",
    "payment",
    "payment details",
    "totals",
    "notes",
    "remarks",
}
_PAYMENT_DATE_LABELS = {"payment date", "paid date", "date paid"}
_PAYMENT_STATUS_LABELS = {"paid status", "payment status", "status"}
_PAYMENT_METHOD_LABELS = {
    "payment method",
    "method of payment",
    "payment term",
    "terms of payment",
}
_EMAIL_LABELS = {"email", "e-mail", "mail", "e - mail"}
_PHONE_LABELS = {"telephone", "phone", "tel", "mobile", "phone number", "telephone number"}


@dataclass(frozen=True)
class ClaimVerificationIssue:
    code: str
    path: str
    message: str
    value: str | None = None
    evidence_ids: tuple[str, ...] = ()

    def prompt_payload(self) -> dict:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "value": self.value,
            "evidence_ids": list(self.evidence_ids),
        }


def trusted_item_claims(evidence: EvidenceDocument) -> tuple[int | None, list[FieldClaim]]:
    """Create item claims directly from locally inferred table cells.

    Table structure and evidence IDs are local facts. A cell is promoted only
    when its semantic role is known and its value is unambiguous; missing cells
    remain available for semantic extraction by the LLM.
    """
    if not evidence.tables:
        return None, []
    claims: list[FieldClaim] = []
    global_index = 0
    for table in sorted(evidence.tables, key=lambda item: item.page):
        for row in table.rows:
            for cell in row.cells:
                if cell.role not in _ITEM_ROLES or not cell.evidence_ids:
                    continue
                value = cell.text.strip()
                if not value:
                    continue
                if cell.role in _NUMERIC_ITEM_ROLES:
                    values = decimal_values_in_text(value)
                    if len(values) != 1:
                        continue
                claims.append(
                    FieldClaim(
                        path=f"items[{global_index}].{cell.role}",
                        value=value,
                        evidence_ids=list(cell.evidence_ids),
                        ambiguity=None,
                    )
                )
            global_index += 1
    return global_index, claims


def trusted_total_claims(evidence: EvidenceDocument) -> list[FieldClaim]:
    """Ground explicit totals from one unambiguous printed label/value pair.

    Totals are never recomputed here. The association uses only the current
    page's OCR boxes and accepts values directly below or beside exact semantic
    labels; it stores no supplier coordinates or page percentages.
    """
    labels_by_path: dict[str, list[EvidenceSpan]] = {
        path: [
            span for span in evidence.spans if _label_text(span.text) in aliases
        ]
        for path, aliases in _TOTAL_LABELS.items()
    }
    label_ids = {span.id for labels in labels_by_path.values() for span in labels}
    selected: dict[str, tuple[EvidenceSpan, EvidenceSpan, Decimal]] = {}
    for path, labels in labels_by_path.items():
        if len(labels) != 1:
            continue
        label = labels[0]
        candidates = []
        for span in evidence.spans:
            if span.page != label.page or span.id in label_ids:
                continue
            values = decimal_values_in_text(span.text)
            if len(values) != 1:
                continue
            score = _total_value_score(label, span)
            if score is not None:
                candidates.append((score, span, values[0]))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            continue
        best_score = candidates[0][0]
        best = [candidate for candidate in candidates if candidate[0] == best_score]
        if len(best) == 1:
            _, value_span, value = best[0]
            selected[path] = (label, value_span, value)

    # One printed amount cannot authorize two different total roles.
    value_ids = [value_span.id for _, value_span, _ in selected.values()]
    duplicated = {item for item in value_ids if value_ids.count(item) > 1}
    return [
        FieldClaim(
            path=path,
            value=format(value, "f"),
            evidence_ids=[label.id, value_span.id],
            ambiguity=None,
        )
        for path, (label, value_span, value) in selected.items()
        if value_span.id not in duplicated
    ]


def trusted_address_claims(
    evidence: EvidenceDocument,
) -> tuple[bool | None, list[FieldClaim]]:
    """Ground unambiguous side-by-side billing and delivery address blocks.

    This is intentionally narrower than general address parsing. It promotes
    values only when one page contains one explicit billing heading and one
    explicit delivery heading, each column has one postal/city line, and the
    immediately adjacent street and country lines are unique. Other layouts
    remain the semantic parser's responsibility.
    """
    billing_labels = [
        span
        for span in evidence.spans
        if _label_text(span.text) in _BILLING_ADDRESS_LABELS
    ]
    delivery_labels = [
        span
        for span in evidence.spans
        if _label_text(span.text) in _DELIVERY_ADDRESS_LABELS
    ]
    if len(billing_labels) != 1 or len(delivery_labels) != 1:
        return None, []
    billing_label = billing_labels[0]
    delivery_label = delivery_labels[0]
    if billing_label.page != delivery_label.page:
        return None, []

    billing_center = _center_x(billing_label)
    delivery_center = _center_x(delivery_label)
    if billing_center == delivery_center:
        return None, []
    left_label, right_label = sorted(
        (billing_label, delivery_label), key=_center_x
    )
    divider = (_center_x(left_label) + _center_x(right_label)) / 2
    boundary = _address_section_boundary(
        evidence, max(billing_label.bbox[3], delivery_label.bbox[3]), billing_label.page
    )

    def column_spans(label: EvidenceSpan) -> list[EvidenceSpan]:
        label_is_left = _center_x(label) < divider
        return sorted(
            (
                span
                for span in evidence.spans
                if span.page == label.page
                and span.bbox[1] >= label.bbox[3]
                and span.bbox[1] < boundary
                and ((_center_x(span) < divider) == label_is_left)
                and _label_text(span.text)
                not in (_BILLING_ADDRESS_LABELS | _DELIVERY_ADDRESS_LABELS)
            ),
            key=lambda span: (_center_y(span), span.bbox[0]),
        )

    billing = _claims_from_address_column("billing_address", column_spans(billing_label))
    delivery = _claims_from_address_column(
        "delivery_address", column_spans(delivery_label)
    )
    if len(billing) != 4 or len(delivery) != 4:
        return True, []
    return True, [*billing, *delivery]


def trusted_company_claims(evidence: EvidenceDocument) -> list[FieldClaim]:
    """Ground one company value attached to one explicit company label."""
    labels = [
        span for span in evidence.spans if _label_text(span.text) in _COMPANY_LABELS
    ]
    if len(labels) != 1:
        return []
    label = labels[0]
    candidates = []
    for span in evidence.spans:
        if span.id == label.id or span.page != label.page or not span.text.strip():
            continue
        score = _text_label_value_score(label, span)
        if score is not None:
            candidates.append((score, span))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return []
    best_score = candidates[0][0]
    best = [span for score, span in candidates if score == best_score]
    if len(best) != 1:
        return []
    value = best[0]
    return [
        FieldClaim(
            path="debtor.company",
            value=value.text.strip(),
            evidence_ids=[label.id, value.id],
            ambiguity=None,
        )
    ]


def trusted_contact_name_claims(evidence: EvidenceDocument) -> list[FieldClaim]:
    """Ground first and last name attached to an explicit contact name label."""
    labels = [
        span
        for span in evidence.spans
        if _label_text(span.text) in _CONTACT_NAME_LABELS
    ]
    if len(labels) != 1:
        return []
    label = labels[0]
    candidates = []
    for span in evidence.spans:
        if span.id == label.id or span.page != label.page or not span.text.strip():
            continue
        score = _text_label_value_score(label, span)
        if score is not None:
            candidates.append((score, span))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return []
    best_score = candidates[0][0]
    best = [span for score, span in candidates if score == best_score]
    if len(best) != 1:
        return []
    value = best[0]
    parts = value.text.strip().split(maxsplit=1)
    if len(parts) == 2:
        first, last = parts
        return [
            FieldClaim(
                path="debtor.first_name",
                value=first.strip(),
                evidence_ids=[label.id, value.id],
                ambiguity=None,
            ),
            FieldClaim(
                path="debtor.last_name",
                value=last.strip(),
                evidence_ids=[label.id, value.id],
                ambiguity=None,
            ),
        ]
    elif len(parts) == 1:
        return [
            FieldClaim(
                path="debtor.last_name",
                value=parts[0].strip(),
                evidence_ids=[label.id, value.id],
                ambiguity=None,
            ),
        ]
    return []


def trusted_contact_details_claims(evidence: EvidenceDocument) -> list[FieldClaim]:
    """Ground email and telephone from explicit labels."""
    claims: list[FieldClaim] = []

    # Email
    email_labels = [
        span for span in evidence.spans if _label_text(span.text) in _EMAIL_LABELS
    ]
    if len(email_labels) == 1:
        label = email_labels[0]
        candidates = []
        for span in evidence.spans:
            if span.id == label.id or span.page != label.page or not span.text.strip():
                continue
            score = _text_label_value_score(label, span)
            if score is not None and "@" in span.text:
                candidates.append((score, span))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            best_score = candidates[0][0]
            best = [span for score, span in candidates if score == best_score]
            if len(best) == 1:
                claims.append(
                    FieldClaim(
                        path="debtor.billing_address.email",
                        value=best[0].text.strip(),
                        evidence_ids=[label.id, best[0].id],
                        ambiguity=None,
                    )
                )

    # Telephone
    phone_labels = [
        span for span in evidence.spans if _label_text(span.text) in _PHONE_LABELS
    ]
    if len(phone_labels) == 1:
        label = phone_labels[0]
        candidates = []
        for span in evidence.spans:
            if span.id == label.id or span.page != label.page or not span.text.strip():
                continue
            score = _text_label_value_score(label, span)
            if score is not None and re.search(r"\d{3,}", span.text):
                candidates.append((score, span))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            best_score = candidates[0][0]
            best = [span for score, span in candidates if score == best_score]
            if len(best) == 1:
                claims.append(
                    FieldClaim(
                        path="debtor.billing_address.telephone",
                        value=best[0].text.strip(),
                        evidence_ids=[label.id, best[0].id],
                        ambiguity=None,
                    )
                )

    return claims


def trusted_payment_claims(evidence: EvidenceDocument) -> list[FieldClaim]:
    """Ground payment method, payment status, and payment date from explicit labels."""
    claims: list[FieldClaim] = []

    # 1. Payment Date
    date_labels = [
        span for span in evidence.spans if _label_text(span.text) in _PAYMENT_DATE_LABELS
    ]
    if len(date_labels) == 1:
        label = date_labels[0]
        candidates = []
        for span in evidence.spans:
            if span.id == label.id or span.page != label.page or not span.text.strip():
                continue
            score = _text_label_value_score(label, span)
            if score is not None and re.search(r"\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4}", span.text):
                candidates.append((score, span))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            best_score = candidates[0][0]
            best = [span for score, span in candidates if score == best_score]
            if len(best) == 1:
                claims.append(
                    FieldClaim(
                        path="payment.payment_date",
                        value=best[0].text.strip(),
                        evidence_ids=[label.id, best[0].id],
                        ambiguity=None,
                    )
                )

    # 2. Payment Status
    status_labels = [
        span for span in evidence.spans if _label_text(span.text) in _PAYMENT_STATUS_LABELS
    ]
    if len(status_labels) == 1:
        label = status_labels[0]
        candidates = []
        for span in evidence.spans:
            if span.id == label.id or span.page != label.page or not span.text.strip():
                continue
            score = _text_label_value_score(label, span)
            if score is not None and _label_text(span.text).upper() in {"PAID", "UNPAID", "OVERDUE", "REFUNDED"}:
                candidates.append((score, span))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            best_score = candidates[0][0]
            best = [span for score, span in candidates if score == best_score]
            if len(best) == 1:
                claims.append(
                    FieldClaim(
                        path="payment.status",
                        value=_label_text(best[0].text).upper(),
                        evidence_ids=[label.id, best[0].id],
                        ambiguity=None,
                    )
                )

    # 3. Payment Method
    method_labels = [
        span for span in evidence.spans if _label_text(span.text) in _PAYMENT_METHOD_LABELS
    ]
    if len(method_labels) == 1:
        label = method_labels[0]
        candidates = []
        for span in evidence.spans:
            if span.id == label.id or span.page != label.page or not span.text.strip():
                continue
            score = _text_label_value_score(label, span)
            if score is not None:
                candidates.append((score, span))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            best_score = candidates[0][0]
            best = [span for score, span in candidates if score == best_score]
            if len(best) == 1:
                claims.append(
                    FieldClaim(
                        path="payment.method",
                        value=best[0].text.strip(),
                        evidence_ids=[label.id, best[0].id],
                        ambiguity=None,
                    )
                )

    return claims


def merge_trusted_debtor_claims(
    response: CompactExtractionClaims,
    delivery_present: bool | None,
    trusted_claims: list[FieldClaim],
) -> CompactExtractionClaims:
    trusted_by_path = {claim.path: claim for claim in trusted_claims}
    merged = [
        claim
        for claim in response.claims
        if _claim_alias(claim.path) not in trusted_by_path
    ]
    merged.extend(trusted_claims)
    updates: dict[str, object] = {"claims": merged}
    if delivery_present is not None:
        updates["delivery_address_present"] = delivery_present
    return response.model_copy(update=updates)


def merge_trusted_item_claims(
    response: CompactExtractionClaims,
    trusted_count: int | None,
    trusted_claims: list[FieldClaim],
) -> CompactExtractionClaims:
    if trusted_count is None:
        return response
    trusted_by_path = {claim.path: claim for claim in trusted_claims}
    merged = [
        claim
        for claim in response.claims
        if claim.path not in trusted_by_path
        and not _item_index_outside(claim.path, trusted_count)
    ]
    merged.extend(trusted_claims)
    return response.model_copy(
        update={"item_count": trusted_count, "claims": merged}
    )


def merge_trusted_total_claims(
    response: CompactExtractionClaims, trusted_claims: list[FieldClaim]
) -> CompactExtractionClaims:
    trusted_by_path = {claim.path: claim for claim in trusted_claims}
    if not trusted_by_path:
        return response
    merged = [
        claim
        for claim in response.claims
        if _claim_alias(claim.path) not in trusted_by_path
    ]
    merged.extend(trusted_claims)
    return response.model_copy(update={"claims": merged})


def verify_claims(
    response: CompactExtractionClaims, evidence: EvidenceDocument
) -> list[ClaimVerificationIssue]:
    issues: list[ClaimVerificationIssue] = []
    spans = evidence.span_index()
    cells = evidence.table_cell_index()
    seen: set[str] = set()

    for claim in response.claims:
        path = _claim_alias(claim.path)
        if path in seen:
            issues.append(_issue("DUPLICATE_CLAIM", claim, "duplicate claim path"))
            continue
        seen.add(path)
        if not _path_allowed(path, response):
            issues.append(_issue("UNSUPPORTED_CLAIM_PATH", claim, "path is not allowlisted"))
            continue
        if claim.value is None:
            continue
        unknown = tuple(item for item in claim.evidence_ids if item not in spans)
        if unknown:
            issues.append(
                _issue(
                    "UNKNOWN_EVIDENCE_ID",
                    claim,
                    "unknown OCR evidence IDs: " + ", ".join(unknown),
                )
            )
            continue
        if not claim.evidence_ids:
            issues.append(_issue("MISSING_EVIDENCE", claim, "value has no OCR evidence IDs"))
            continue

        item_match = re.fullmatch(r"items\[(\d+)]\.([a-z_]+)", path)
        if item_match and cells:
            expected_row = int(item_match.group(1))
            expected_role = item_match.group(2)
            mapped = [cells[item] for item in claim.evidence_ids if item in cells]
            if not mapped or any(
                row != expected_row or role != expected_role for row, role in mapped
            ):
                issues.append(
                    _issue(
                        "TABLE_CELL_MISMATCH",
                        claim,
                        f"evidence must come from item[{expected_row}].{expected_role}",
                    )
                )
                continue

        claim_spans = [spans[item] for item in claim.evidence_ids]
        if not _claim_value_supported(path, str(claim.value), claim_spans):
            issues.append(
                _issue(
                    "UNGROUNDED_VALUE",
                    claim,
                    "claim value conflicts with its referenced OCR text",
                )
            )

    claims_by_path = {_claim_alias(claim.path): claim for claim in response.claims}
    for path in sorted(_REQUIRED_PATHS):
        if not _has_value(claims_by_path.get(path)):
            issues.append(
                ClaimVerificationIssue(
                    code="MISSING_REQUIRED_CLAIM",
                    path=path,
                    message="required source field was omitted or null",
                )
            )
    if not any(
        _has_value(claims_by_path.get(path))
        for path in ("debtor.company", "debtor.first_name", "debtor.last_name")
    ):
        issues.append(
            ClaimVerificationIssue(
                code="MISSING_IDENTITY_CLAIM",
                path="debtor.company",
                message="extract a company or supported personal contact name",
            )
        )
    if response.item_count == 0:
        issues.append(
            ClaimVerificationIssue(
                code="MISSING_ITEM_ROWS",
                path="item_count",
                message="purchase order contains no extracted item rows",
            )
        )
    for item_index in range(response.item_count):
        for field in sorted(_ITEM_ROLES):
            path = f"items[{item_index}].{field}"
            if not _has_value(claims_by_path.get(path)):
                issues.append(
                    ClaimVerificationIssue(
                        code="MISSING_REQUIRED_CLAIM",
                        path=path,
                        message="required item cell was omitted or null",
                    )
                )
    if response.delivery_address_present:
        for field in ("street", "zip", "city", "country"):
            path = f"debtor.delivery_address.{field}"
            if not _has_value(claims_by_path.get(path)):
                issues.append(
                    ClaimVerificationIssue(
                        code="MISSING_REQUIRED_CLAIM",
                        path=path,
                        message="explicit delivery address field was omitted or null",
                    )
                )
    status = claims_by_path.get("payment.status")
    payment_date = claims_by_path.get("payment.payment_date")
    if status and str(status.value or "").strip().upper() == "PAID" and not (
        payment_date and str(payment_date.value or "").strip()
    ):
        issues.append(
            ClaimVerificationIssue(
                code="MISSING_CONDITIONAL_CLAIM",
                path="payment.payment_date",
                message="source status PAID requires extraction of a printed payment date",
            )
        )
    return issues


def apply_focused_repair(
    original: CompactExtractionClaims,
    repaired: CompactExtractionClaims,
    issues: list[ClaimVerificationIssue],
) -> CompactExtractionClaims:
    """Accept changes only to fields that deterministic verification rejected."""
    paths = {issue.path for issue in issues}
    repairable_paths = set(paths)
    if "debtor.company" in paths:
        repairable_paths.update(
            {"debtor.company", "debtor.first_name", "debtor.last_name"}
        )
    repaired_item_count = (
        repaired.item_count if "item_count" in paths else original.item_count
    )
    if "item_count" in paths:
        repairable_paths.update(
            _claim_alias(claim.path)
            for claim in repaired.claims
            if re.fullmatch(r"items\[\d+]\.[a-z_]+", claim.path)
        )
    repaired_by_path = {_claim_alias(claim.path): claim for claim in repaired.claims}
    retained = [
        claim
        for claim in original.claims
        if _claim_alias(claim.path) not in repairable_paths
    ]
    retained.extend(
        repaired_by_path[path]
        for path in repairable_paths
        if path in repaired_by_path
    )
    return original.model_copy(
        update={"item_count": repaired_item_count, "claims": retained}
    )


def sanitize_unverified_claims(
    response: CompactExtractionClaims, issues: list[ClaimVerificationIssue]
) -> CompactExtractionClaims:
    """Turn unresolved claims into explicit ambiguity so review can be persisted."""
    messages: dict[str, list[str]] = {}
    for issue in issues:
        messages.setdefault(issue.path, []).append(f"{issue.code}: {issue.message}")
    cleaned: list[FieldClaim] = []
    seen: set[str] = set()
    for claim in response.claims:
        path = _claim_alias(claim.path)
        if path in seen or not _path_allowed(path, response):
            continue
        seen.add(path)
        if path in messages:
            cleaned.append(
                claim.model_copy(
                    update={
                        "path": path,
                        "value": None,
                        "ambiguity": "; ".join(messages[path]),
                    }
                )
            )
        else:
            cleaned.append(claim.model_copy(update={"path": path}))
    for path, reasons in messages.items():
        if path not in seen and _path_allowed(path, response):
            cleaned.append(
                FieldClaim(
                    path=path,
                    value=None,
                    evidence_ids=[],
                    ambiguity="; ".join(reasons),
                )
            )
    return response.model_copy(update={"claims": cleaned})


def relevant_evidence_payload(
    evidence: EvidenceDocument, issues: list[ClaimVerificationIssue]
) -> dict:
    wanted = {item for issue in issues for item in issue.evidence_ids}
    ordered = list(evidence.spans)
    positions = {span.id: index for index, span in enumerate(ordered)}
    for evidence_id in tuple(wanted):
        position = positions.get(evidence_id)
        if position is None:
            continue
        for span in ordered[max(0, position - 2) : position + 3]:
            wanted.add(span.id)
    selected = [span for span in ordered if span.id in wanted]
    missing_paths = [issue.path for issue in issues if not issue.evidence_ids]
    for path in missing_paths:
        item = re.fullmatch(r"items\[(\d+)]\.([a-z_]+)", path)
        if item:
            expected = (int(item.group(1)), item.group(2))
            for evidence_id, location in evidence.table_cell_index().items():
                if location == expected:
                    wanted.add(evidence_id)
            continue
        terms = _repair_search_terms(path)
        for position, span in enumerate(ordered):
            if any(term in _normalize(span.text).casefold() for term in terms):
                for nearby in ordered[max(0, position - 3) : position + 9]:
                    wanted.add(nearby.id)
    selected = [span for span in ordered if span.id in wanted]
    if not selected:
        selected = ordered[:32]
    return {
        "span_columns": ["id", "text", "confidence", "bbox", "page"],
        "spans": [
            [span.id, span.text, round(span.confidence, 3), list(span.bbox), span.page]
            for span in selected
        ],
    }


def normalize_optional_placeholders(response: CompactExtractionClaims) -> CompactExtractionClaims:
    claims = []
    for claim in response.claims:
        path = _claim_alias(claim.path)
        optional = path in _OPTIONAL_PLACEHOLDER_PATHS or bool(
            re.fullmatch(
                r"debtor\.(?:billing_address|delivery_address)\."
                r"(?:email|telephone|additional_name|address_specification|district)",
                path,
            )
        )
        if optional and str(claim.value or "").strip().casefold() in _PLACEHOLDERS:
            claim = claim.model_copy(update={"path": path, "value": None})
        claims.append(claim)
    return response.model_copy(update={"claims": claims})


def normalize_proven_ambiguities(
    response: CompactExtractionClaims, evidence: EvidenceDocument
) -> CompactExtractionClaims:
    """Remove an LLM caveat only when local evidence proves one exact meaning."""
    index = evidence.span_index()
    claims = []
    allowed_statuses = {"PAID", "UNPAID", "OVERDUE", "PARTIALLY PAID", "REFUNDED"}
    for claim in response.claims:
        if not claim.ambiguity or claim.value is None:
            claims.append(claim)
            continue
        if claim.path == "payment.payment_date" and _labeled_payment_date_is_proven(
            claim, evidence
        ):
            claims.append(claim.model_copy(update={"ambiguity": None}))
            continue
        if claim.path != "payment.status":
            claims.append(claim)
            continue
        expected = _normalize(str(claim.value)).upper()
        cited = [index[item] for item in claim.evidence_ids if item in index]
        exact_statuses = {
            _normalize(span.text).upper()
            for span in cited
            if _normalize(span.text).upper() in allowed_statuses
        }
        if exact_statuses == {expected}:
            claim = claim.model_copy(update={"ambiguity": None})
        claims.append(claim)
    return response.model_copy(update={"claims": claims})


def _labeled_payment_date_is_proven(
    claim: FieldClaim, evidence: EvidenceDocument
) -> bool:
    expected = _date_value(str(claim.value or ""))
    if expected is None:
        return False
    index = evidence.span_index()
    cited_values = [
        index[evidence_id]
        for evidence_id in claim.evidence_ids
        if evidence_id in index and expected in _dates_in_text(index[evidence_id].text)
    ]
    if len(cited_values) != 1:
        return False
    value_span = cited_values[0]
    labels = [
        span
        for span in evidence.spans
        if span.page == value_span.page
        and _label_text(span.text) in _PAYMENT_DATE_LABELS
        and _total_value_score(span, value_span) is not None
    ]
    return len(labels) == 1


def normalize_delivery_presence(
    response: CompactExtractionClaims, evidence: EvidenceDocument
) -> CompactExtractionClaims:
    """Interpret an explicit no-delivery marker without treating headings as data."""
    if not response.delivery_address_present:
        return response
    ordered = list(evidence.spans)
    absent = False
    for position, span in enumerate(ordered):
        if not re.search(r"\b(?:delivery|shipping)\s+address\b", span.text, re.IGNORECASE):
            continue
        nearby = " ".join(
            item.text for item in ordered[position + 1 : position + 4]
        )
        if re.search(
            r"(?:\bno\s+(?:delivery|shipping)\b|\bnot\s+(?:applicable|shipped)\b|"
            r"\bbefore\s+shipment\b|\bn/?a\b|^\s*[—–-])",
            nearby,
            re.IGNORECASE,
        ):
            absent = True
            break
    if not absent:
        return response
    return response.model_copy(
        update={
            "delivery_address_present": False,
            "claims": [
                claim
                for claim in response.claims
                if not claim.path.startswith("debtor.delivery_address.")
            ],
        }
    )


def _issue(code: str, claim: FieldClaim, message: str) -> ClaimVerificationIssue:
    return ClaimVerificationIssue(
        code=code,
        path=_claim_alias(claim.path),
        message=message,
        value=claim.value,
        evidence_ids=tuple(claim.evidence_ids),
    )


def _has_value(claim: FieldClaim | None) -> bool:
    return bool(claim and claim.value is not None and str(claim.value).strip())


def _repair_search_terms(path: str) -> tuple[str, ...]:
    if path.startswith("debtor.billing_address"):
        return ("billing address", "bill to", "addresses")
    if path.startswith("debtor.delivery_address"):
        return ("delivery address", "ship to", "addresses")
    if path.startswith("debtor."):
        return ("customer", "contact", "company", "buyer")
    if path.startswith("totals.total_net"):
        return ("net total", "total net", "subtotal")
    if path.startswith("totals.total_vat"):
        return ("vat total", "total vat", "tax total")
    if path.startswith("totals.total_gross"):
        return ("gross total", "total gross", "grand total")
    if path.startswith("payment"):
        return ("payment", "paid", "status", "due date")
    if path == "order_date":
        return ("order date", "date")
    if path == "external_reference":
        return ("reference", "order no", "order number", "ref")
    if path == "currency":
        return ("currency", " eur", " usd", " gbp", " jpy")
    if path == "item_count":
        return ("items", "sku", "quantity", "qty")
    return tuple(part for part in re.split(r"[._]", path) if len(part) > 2)


def _label_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    normalized = re.sub(r"[^\w]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _total_value_score(
    label: EvidenceSpan, value: EvidenceSpan
) -> tuple[int, int, int] | None:
    label_left, label_top, label_right, label_bottom = label.bbox
    value_left, value_top, value_right, value_bottom = value.bbox
    label_height = max(1, label_bottom - label_top)
    value_height = max(1, value_bottom - value_top)
    horizontal_overlap = min(label_right, value_right) - max(label_left, value_left)
    vertical_overlap = min(label_bottom, value_bottom) - max(label_top, value_top)
    label_center_x = (label_left + label_right) // 2
    value_center_x = (value_left + value_right) // 2
    label_center_y = (label_top + label_bottom) // 2
    value_center_y = (value_top + value_bottom) // 2

    below_gap = value_top - label_bottom
    if horizontal_overlap > 0 and 0 <= below_gap <= 8 * max(label_height, value_height):
        return (
            below_gap + abs(value_center_x - label_center_x),
            0,
            below_gap,
        )

    right_gap = value_left - label_right
    if vertical_overlap > 0 and 0 <= right_gap <= 20 * max(label_height, value_height):
        return (
            right_gap + abs(value_center_y - label_center_y),
            1,
            right_gap,
        )
    return None


def _text_label_value_score(
    label: EvidenceSpan, value: EvidenceSpan
) -> tuple[int, int, int] | None:
    """Associate a form label with its nearest row before considering alignment."""
    label_left, label_top, label_right, label_bottom = label.bbox
    value_left, value_top, value_right, value_bottom = value.bbox
    label_height = max(1, label_bottom - label_top)
    value_height = max(1, value_bottom - value_top)
    horizontal_overlap = min(label_right, value_right) - max(label_left, value_left)
    vertical_overlap = min(label_bottom, value_bottom) - max(label_top, value_top)
    label_center_x = (label_left + label_right) // 2
    value_center_x = (value_left + value_right) // 2
    label_center_y = (label_top + label_bottom) // 2
    value_center_y = (value_top + value_bottom) // 2

    below_gap = value_top - label_bottom
    if horizontal_overlap > 0 and 0 <= below_gap <= 4 * max(label_height, value_height):
        return (0, below_gap, abs(value_center_x - label_center_x))

    right_gap = value_left - label_right
    if vertical_overlap > 0 and 0 <= right_gap <= 12 * max(label_height, value_height):
        return (1, right_gap, abs(value_center_y - label_center_y))
    return None


def _claims_from_address_column(
    kind: str, spans: list[EvidenceSpan]
) -> list[FieldClaim]:
    postal_candidates: list[tuple[EvidenceSpan, str, str]] = []
    for span in spans:
        parsed = _split_unambiguous_postal_city(span.text)
        if parsed is not None:
            postal_candidates.append((span, *parsed))
    if len(postal_candidates) != 1:
        return []
    postal_span, postal, city = postal_candidates[0]
    postal_y = _center_y(postal_span)
    before = [span for span in spans if _center_y(span) < postal_y]
    after = [span for span in spans if _center_y(span) > postal_y]
    if not before or not after:
        return []

    street_span = max(before, key=_center_y)
    country_span = min(after, key=_center_y)
    country = country_span.text.strip()
    if not re.search(r"[^\W\d_]", country, re.UNICODE) or re.search(r"\d", country):
        return []
    # Reject implausibly distant neighbors rather than reaching into another
    # section and assigning an unrelated line as an address component.
    line_height = max(1, postal_span.bbox[3] - postal_span.bbox[1])
    if postal_span.bbox[1] - street_span.bbox[3] > 4 * line_height:
        return []
    if country_span.bbox[1] - postal_span.bbox[3] > 4 * line_height:
        return []

    prefix = f"debtor.{kind}"
    return [
        FieldClaim(
            path=f"{prefix}.street",
            value=street_span.text.strip(),
            evidence_ids=[street_span.id],
            ambiguity=None,
        ),
        FieldClaim(
            path=f"{prefix}.zip",
            value=postal,
            evidence_ids=[postal_span.id],
            ambiguity=None,
        ),
        FieldClaim(
            path=f"{prefix}.city",
            value=city,
            evidence_ids=[postal_span.id],
            ambiguity=None,
        ),
        FieldClaim(
            path=f"{prefix}.country",
            value=country,
            evidence_ids=[country_span.id],
            ambiguity=None,
        ),
    ]


def _split_unambiguous_postal_city(value: str) -> tuple[str, str] | None:
    source = _normalize(value).strip(" ,")
    patterns = (
        re.compile(
            r"^(?P<postal>\d{4,6}(?:[- ]\d{3,4})?)\s+(?P<city>.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?P<city>.+?)[,\s]+(?P<postal>\d{4,6}(?:-\d{3,4})?)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?P<city>.+?)[,\s]+(?P<postal>[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?P<city>.+?)[,\s]+(?P<postal>[A-Z]\d[A-Z]\s?\d[A-Z]\d)$",
            re.IGNORECASE,
        ),
    )
    matches = []
    for pattern in patterns:
        match = pattern.fullmatch(source)
        if match is None:
            continue
        postal = match.group("postal").strip().upper()
        city = match.group("city").strip(" ,")
        if any(character.isdigit() for character in postal) and any(
            character.isalpha() for character in city
        ):
            matches.append((postal, city))
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _address_section_boundary(
    evidence: EvidenceDocument, heading_bottom: int, page: int
) -> int:
    candidates = [
        span.bbox[1]
        for span in evidence.spans
        if span.page == page
        and span.bbox[1] > heading_bottom
        and _label_text(span.text) in _ADDRESS_SECTION_BOUNDARIES
    ]
    page_height = next(
        (item.height for item in evidence.pages if item.page == page),
        max((span.bbox[3] for span in evidence.spans if span.page == page), default=0)
        + 1,
    )
    return min(candidates, default=page_height)


def _center_x(span: EvidenceSpan) -> float:
    return (span.bbox[0] + span.bbox[2]) / 2


def _center_y(span: EvidenceSpan) -> float:
    return (span.bbox[1] + span.bbox[3]) / 2


def _claim_alias(path: str) -> str:
    return {
        "debtor.email": "debtor.billing_address.email",
        "debtor.telephone": "debtor.billing_address.telephone",
    }.get(path, path)


def _path_allowed(path: str, response: CompactExtractionClaims) -> bool:
    if path in _ROOT_PATHS:
        return True
    address = re.fullmatch(r"debtor\.(billing_address|delivery_address)\.([a-z_]+)", path)
    if address:
        kind, field = address.groups()
        return field in _ADDRESS_FIELDS and (
            kind != "delivery_address" or response.delivery_address_present
        )
    item = re.fullmatch(r"items\[(\d+)]\.([a-z_]+)", path)
    return bool(
        item
        and int(item.group(1)) < response.item_count
        and item.group(2) in _ITEM_ROLES
    )


def _item_index_outside(path: str, item_count: int) -> bool:
    match = re.fullmatch(r"items\[(\d+)]\.[a-z_]+", path)
    return bool(match and int(match.group(1)) >= item_count)


def _claim_value_supported(path: str, value: str, spans: list[EvidenceSpan]) -> bool:
    source = " ".join(span.text for span in spans)
    if re.search(r"(?:quantity|unit_net_price|vat_percent|discount_percent|source_total|total_net|total_vat|total_gross)$", path):
        expected = parse_decimal_text(value)
        return expected is not None and expected in decimal_values_in_text(source)
    if path in {"order_date", "payment.payment_date"}:
        expected = _date_value(value)
        return expected is not None and expected in _dates_in_text(source)
    if path == "payment.status":
        expected = _normalize(value).upper()
        source_statuses = {
            match.group(0).upper().replace("  ", " ")
            for match in re.finditer(
                r"\b(?:PARTIALLY\s+PAID|UNPAID|OVERDUE|REFUNDED|PAID)\b",
                _normalize(source),
                re.IGNORECASE,
            )
        }
        return expected in source_statuses
    if path == "currency":
        expected = value.strip().upper()
        aliases = {
            "EUR": ("eur", "€", "euro"),
            "USD": ("usd", "$", "dollar"),
            "GBP": ("gbp", "£", "pound"),
        }.get(expected, (expected.casefold(),))
        return any(alias in unicodedata.normalize("NFKC", source).casefold() for alias in aliases)
    normalized = _normalize(value)
    joined = {_normalize(source), _normalize("".join(span.text for span in spans))}
    return bool(normalized) and any(normalized in candidate for candidate in joined)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip())


def _date_value(value: str) -> date | None:
    for format_string in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), format_string).date()
        except ValueError:
            pass
    return None


def _dates_in_text(text: str) -> set[date]:
    values = re.findall(r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b", text)
    return {parsed for value in values if (parsed := _date_value(value)) is not None}
