from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from fakturama_automation.domain.models import FieldEvidence, OrderInput

from .ocr import OcrLine, OcrResult


@dataclass(frozen=True)
class _ItemRow:
    sku: OcrLine
    description: OcrLine
    quantity: OcrLine
    unit_price: OcrLine
    discount: OcrLine
    vat: OcrLine
    total: OcrLine


class SpatialOrderParser:
    """Fail-closed parser for labelled sales-order layouts.

    It uses label anchors and normalized coordinates. Unknown layouts should be
    routed to the optional text LLM instead of expanding these rules by guesswork.
    """

    uses_image = False
    requires_review = False

    def parse(self, ocr: OcrResult, image_path: Path | None = None) -> OrderInput:
        width = ocr.quality.width if ocr.quality else _estimated_width(ocr.lines)
        lines = [line for line in ocr.lines if line.text.strip()]

        external_reference = _best_match(lines, r"^[A-Z]+-\d{4}-[A-Z0-9-]+$")
        dates = sorted(
            _matches(lines, r"^\d{4}-\d{2}-\d{2}$"),
            key=lambda line: _center(line)[1],
        )
        if len(dates) < 2:
            raise ValueError("spatial parser requires order and payment dates")

        company = _best_containing(lines, ("gmbh", "ltd", "llc", "inc", "corp"))
        customer_top = _label_y(lines, "customer and contact")
        address_top = _label_y(lines, "addresses")
        customer_lines = [
            line
            for line in lines
            if customer_top < _center(line)[1] < address_top
        ]
        alias = _best_match(
            customer_lines, r"^[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+$"
        )
        email = _best_match(lines, r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
        phone = _best_match(lines, r"^\+[\d][\d\s().-]{6,}$")

        payment_top = _label_y(lines, "payment")
        billing_lines = _section_column(
            lines, address_top, payment_top, 0, width / 2
        )
        delivery_lines = _section_column(
            lines, address_top, payment_top, width / 2, width
        )
        billing = _parse_address(billing_lines, company.text, email.text, phone.text)
        delivery_company = _best_containing(
            delivery_lines, ("warehouse", "gmbh", "ltd", "llc", "inc", "corp")
        )
        delivery = _parse_address(delivery_lines, delivery_company.text, None, None)

        payment_bottom = _label_y(lines, "items")
        payment_lines = [
            line
            for line in lines
            if payment_top < _center(line)[1] < payment_bottom
        ]
        payment_method = _best_containing(
            payment_lines,
            ("bank transfer", "credit card", "direct debit", "cash", "paypal"),
        )
        paid_line = _best_exact(payment_lines, ("paid", "unpaid"))

        rows = _parse_item_rows(lines, width, payment_bottom, _totals_y(lines))
        total_lines = sorted(
            [
                line
                for line in lines
                if _center(line)[1] > _totals_y(lines)
                and re.fullmatch(r"[A-Z]{3}\s+\d+[.,]\d{2}", line.text.strip())
            ],
            key=lambda line: _center(line)[0],
        )
        if len(total_lines) < 3:
            raise ValueError("spatial parser could not identify net, VAT, and gross totals")

        evidence: dict[str, FieldEvidence] = {}
        _add_evidence(evidence, "external_reference", external_reference)
        _add_evidence(evidence, "order_date", dates[0])
        _add_evidence(evidence, "debtor.company", company)
        _add_evidence(evidence, "debtor.billing_address.zip", billing["zip_line"])
        _add_evidence(evidence, "debtor.billing_address.city", billing["zip_line"])
        _add_evidence(evidence, "payment.method", payment_method)
        _add_evidence(evidence, "payment.status", paid_line)
        _add_evidence(evidence, "payment.payment_date", dates[-1])

        items: list[dict] = []
        for index, row in enumerate(rows):
            items.append(
                {
                    "sku": row.sku.text,
                    "description": row.description.text,
                    "quantity": _number(row.quantity.text),
                    "unit_net_price": _number(row.unit_price.text),
                    "vat_percent": _number(row.vat.text),
                    "discount_percent": _number(row.discount.text),
                    "source_total": _number(row.total.text),
                }
            )
            for name, line in (
                ("sku", row.sku),
                ("quantity", row.quantity),
                ("unit_net_price", row.unit_price),
                ("vat_percent", row.vat),
                ("discount_percent", row.discount),
                ("source_total", row.total),
            ):
                _add_evidence(evidence, f"items[{index}].{name}", line)

        for name, line in zip(
            ("total_net", "total_vat", "total_gross"), total_lines[:3], strict=True
        ):
            _add_evidence(evidence, f"totals.{name}", line)

        currency = total_lines[0].text.split()[0]
        return OrderInput.model_validate(
            {
                "order_date": dates[0].text,
                "external_reference": external_reference.text,
                "debtor": {
                    "company": company.text,
                    "alias": alias.text,
                    "billing_address": billing["address"],
                    "delivery_address": delivery["address"],
                },
                "payment": {
                    "method": payment_method.text,
                    "status": paid_line.text.upper(),
                    "payment_date": dates[-1].text
                    if paid_line.text.casefold() == "paid"
                    else None,
                },
                "items": items,
                "totals": {
                    "total_net": _number(total_lines[0].text),
                    "total_vat": _number(total_lines[1].text),
                    "total_gross": _number(total_lines[2].text),
                },
                "currency": currency,
                "evidence": evidence,
            }
        )


def _parse_item_rows(
    lines: list[OcrLine], width: int, top: int, bottom: int
) -> list[_ItemRow]:
    table_lines = [line for line in lines if top < _center(line)[1] < bottom]
    sku_lines = [
        line
        for line in table_lines
        if re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)+", line.text.strip())
        and _center(line)[0] < width * 0.35
    ]
    rows: list[_ItemRow] = []
    for sku in sorted(sku_lines, key=lambda line: _center(line)[1]):
        row_y = _center(sku)[1]
        row = [line for line in table_lines if abs(_center(line)[1] - row_y) <= 8]
        description = _closest_x(row, width * 0.30, exclude={sku})
        rows.append(
            _ItemRow(
                sku=sku,
                description=description,
                quantity=_closest_x(row, width * 0.505, exclude={sku, description}),
                unit_price=_closest_x(row, width * 0.635, exclude={sku, description}),
                discount=_closest_x(row, width * 0.72, exclude={sku, description}),
                vat=_closest_x(row, width * 0.785, exclude={sku, description}),
                total=_closest_x(row, width * 0.87, exclude={sku, description}),
            )
        )
    if not rows:
        raise ValueError("spatial parser could not identify item rows")
    return rows


def _parse_address(
    lines: list[OcrLine], company: str, email: str | None, phone: str | None
) -> dict:
    zip_line = _best_match(lines, r"^\d{4,6}\s+.+$")
    zip_code, city = zip_line.text.split(maxsplit=1)
    country = _best_exact(lines, ("germany", "deutschland", "france", "italy", "spain"))
    street = _best_match(lines, r"^(?!\d{4,6}\s).*[A-Za-z].*\d+[A-Za-z]?$", exclude={zip_line})
    return {
        "zip_line": zip_line,
        "address": {
            "street": street.text,
            "zip": zip_code,
            "city": city,
            "country": country.text,
            "email": email,
            "telephone": phone,
            "additional_name": None,
            "address_specification": None,
            "district": None,
        },
    }


def _section_column(
    lines: list[OcrLine], top: int, bottom: int, left: float, right: float
) -> list[OcrLine]:
    ignored = {"addresses", "billing address", "delivery address"}
    return [
        line
        for line in lines
        if top < _center(line)[1] < bottom
        and left <= _center(line)[0] < right
        and line.text.strip().casefold() not in ignored
    ]


def _matches(lines: list[OcrLine], pattern: str) -> list[OcrLine]:
    expression = re.compile(pattern, re.IGNORECASE)
    return [line for line in lines if expression.fullmatch(line.text.strip())]


def _best_match(
    lines: list[OcrLine], pattern: str, *, exclude: set[OcrLine] | None = None
) -> OcrLine:
    candidates = [line for line in _matches(lines, pattern) if line not in (exclude or set())]
    if not candidates:
        raise ValueError(f"spatial parser could not find value matching {pattern!r}")
    return max(candidates, key=lambda line: line.confidence)


def _best_containing(lines: list[OcrLine], values: tuple[str, ...]) -> OcrLine:
    candidates = [
        line
        for line in lines
        if any(value in line.text.casefold() for value in values)
    ]
    if not candidates:
        raise ValueError(f"spatial parser could not find any of {values}")
    return max(candidates, key=lambda line: line.confidence)


def _best_exact(lines: list[OcrLine], values: tuple[str, ...]) -> OcrLine:
    candidates = [
        line for line in lines if line.text.strip().casefold() in set(values)
    ]
    if not candidates:
        raise ValueError(f"spatial parser could not find exact value in {values}")
    return max(candidates, key=lambda line: line.confidence)


def _closest_x(
    lines: list[OcrLine], target: float, *, exclude: set[OcrLine]
) -> OcrLine:
    candidates = [line for line in lines if line not in exclude]
    if not candidates:
        raise ValueError("spatial parser found an incomplete item row")
    return min(candidates, key=lambda line: abs(_center(line)[0] - target))


def _label_y(lines: list[OcrLine], label: str) -> int:
    candidates = [
        line for line in lines if line.text.strip().casefold() == label.casefold()
    ]
    if not candidates:
        raise ValueError(f"spatial parser requires the {label!r} section")
    return min(_center(line)[1] for line in candidates)


def _totals_y(lines: list[OcrLine]) -> int:
    candidates = [
        line
        for line in lines
        if line.text.strip().casefold() in {"net total", "vat total", "gross total"}
    ]
    if not candidates:
        raise ValueError("spatial parser requires a totals section")
    return min(_center(line)[1] for line in candidates)


def _center(line: OcrLine) -> tuple[float, float]:
    left, top, right, bottom = line.bounding_box
    return ((left + right) / 2, (top + bottom) / 2)


def _estimated_width(lines: list[OcrLine]) -> int:
    return max((line.bounding_box[2] for line in lines), default=1)


def _number(value: str) -> Decimal:
    cleaned = re.sub(r"[^0-9,.-]", "", value).replace(",", ".")
    return Decimal(cleaned)


def _add_evidence(
    evidence: dict[str, FieldEvidence], path: str, line: OcrLine
) -> None:
    evidence[path] = FieldEvidence(
        source_text=line.text,
        confidence=line.confidence,
        bounding_box=line.bounding_box,
    )
