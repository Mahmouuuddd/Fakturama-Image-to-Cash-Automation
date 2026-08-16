from __future__ import annotations

import re
import statistics
import unicodedata

from .evidence import (
    EvidencePage,
    EvidenceSpan,
    EvidenceTable,
    EvidenceTableCell,
    EvidenceTableColumn,
    EvidenceTableRow,
)


_REQUIRED_HEADER_ROLES = {"quantity", "sku", "source_total"}


def infer_item_tables(
    spans: tuple[EvidenceSpan, ...], pages: tuple[EvidencePage, ...]
) -> tuple[EvidenceTable, ...]:
    """Infer item tables from detected headers and relative OCR geometry.

    No supplier coordinates are stored. Every column band is derived from the
    current page's own header positions, and item rows are anchored by its SKU
    cells.
    """
    tables = []
    for page in pages:
        page_spans = [span for span in spans if span.page == page.page]
        table = _infer_page_table(page_spans, page)
        if table is not None:
            tables.append(table)
    return tuple(tables)


def _infer_page_table(
    spans: list[EvidenceSpan], page: EvidencePage
) -> EvidenceTable | None:
    candidates = [
        (span, role)
        for span in spans
        if (role := _header_role(span.text)) is not None
    ]
    if not candidates:
        return None
    heights = [max(1, span.bbox[3] - span.bbox[1]) for span, _ in candidates]
    tolerance = max(12, int(statistics.median(heights) * 1.25))
    header_group = _best_header_group(candidates, tolerance)
    if header_group is None:
        return None

    selected: dict[str, EvidenceSpan] = {}
    anchor_y = statistics.median(_center_y(span) for span, _ in header_group)
    for span, role in header_group:
        current = selected.get(role)
        if current is None or abs(_center_y(span) - anchor_y) < abs(
            _center_y(current) - anchor_y
        ):
            selected[role] = span
    if not _REQUIRED_HEADER_ROLES.issubset(selected):
        return None

    ordered_headers = sorted(selected.items(), key=lambda pair: _center_x(pair[1]))
    centers = [_center_x(span) for _, span in ordered_headers]
    columns = []
    for index, (role, span) in enumerate(ordered_headers):
        left = 0 if index == 0 else int((centers[index - 1] + centers[index]) / 2)
        right = (
            page.width
            if index == len(ordered_headers) - 1
            else int((centers[index] + centers[index + 1]) / 2)
        )
        columns.append(
            EvidenceTableColumn(
                role=role,
                header_ids=(span.id,),
                x_range=(left, right),
            )
        )

    header_bottom = max(span.bbox[3] for span in selected.values())
    table_bottom = _totals_boundary(spans, header_bottom, page.height)
    sku_column = next(column for column in columns if column.role == "sku")
    sku_anchors = [
        span
        for span in spans
        if header_bottom < _center_y(span) < table_bottom
        and _in_column(span, sku_column)
        and span.id not in sku_column.header_ids
    ]
    sku_anchors.sort(key=_center_y)
    if not sku_anchors:
        return None

    rows = []
    anchor_centers = [_center_y(span) for span in sku_anchors]
    for index, sku in enumerate(sku_anchors):
        top = (
            header_bottom
            if index == 0
            else int((anchor_centers[index - 1] + anchor_centers[index]) / 2)
        )
        bottom = (
            table_bottom
            if index == len(sku_anchors) - 1
            else int((anchor_centers[index] + anchor_centers[index + 1]) / 2)
        )
        # ``table_bottom`` is the center line of the first totals label. Keep
        # that boundary exclusive so totals headings cannot contaminate the
        # final item row when their center rounds to the same integer.
        cell_spans = [span for span in spans if top < _center_y(span) < bottom]
        cells = []
        for column in columns:
            values = [span for span in cell_spans if _in_column(span, column)]
            values.sort(key=lambda span: (span.bbox[1], span.bbox[0]))
            if not values:
                continue
            cells.append(
                EvidenceTableCell(
                    role=column.role,
                    text=" ".join(span.text for span in values),
                    evidence_ids=tuple(span.id for span in values),
                )
            )
        rows.append(
            EvidenceTableRow(
                index=index,
                y_range=(top, bottom),
                cells=tuple(cells),
            )
        )

    return EvidenceTable(
        page=page.page,
        header_ids=tuple(span.id for span in selected.values()),
        columns=tuple(columns),
        rows=tuple(rows),
    )


def _best_header_group(
    candidates: list[tuple[EvidenceSpan, str]], tolerance: int
) -> list[tuple[EvidenceSpan, str]] | None:
    best = None
    best_score = (-1, -1)
    for anchor, _ in candidates:
        group = [
            candidate
            for candidate in candidates
            if candidate[0].page == anchor.page
            and abs(_center_y(candidate[0]) - _center_y(anchor)) <= tolerance
        ]
        roles = {role for _, role in group}
        score = (len(roles), sum(role in _REQUIRED_HEADER_ROLES for role in roles))
        if score > best_score:
            best, best_score = group, score
    if best is None or best_score[0] < 4 or best_score[1] < 3:
        return None
    return best


def _header_role(value: str) -> str | None:
    text = _normalize(value)
    if text in {"#", "pos", "position", "row", "zeile"}:
        return "row_number"
    if text in {
        "sku",
        "item no",
        "item number",
        "article no",
        "article number",
        "artikelnummer",
    }:
        return "sku"
    if text in {"description", "item description", "beschreibung"}:
        return "description"
    if text in {"qty", "quantity", "menge", "anzahl"}:
        return "quantity"
    if text in {"unit", "uom", "einheit"}:
        return "unit"
    if (
        ("unit" in text and ("net" in text or "price" in text))
        or text in {"unit price", "price net", "einzelpreis", "e preis"}
    ):
        return "unit_net_price"
    if text in {"disc", "discount", "rabatt"}:
        return "discount_percent"
    if text in {"vat", "tax", "mwst", "ust"}:
        return "vat_percent"
    if text in {
        "line net",
        "line total",
        "net amount",
        "amount",
        "zeilensumme",
        "gesamtpreis",
    }:
        return "source_total"
    return None


def _totals_boundary(spans: list[EvidenceSpan], header_bottom: int, fallback: int) -> int:
    candidates = [
        _center_y(span)
        for span in spans
        if _center_y(span) > header_bottom and _is_totals_label(span.text)
    ]
    return int(min(candidates)) if candidates else fallback


def _is_totals_label(value: str) -> bool:
    text = _normalize(value)
    return text in {
        "totals",
        "net total",
        "total net",
        "vat total",
        "total vat",
        "gross total",
        "total gross",
        "subtotal",
        "summe netto",
        "mwst gesamt",
        "gesamt",
    }


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    normalized = re.sub(r"[^\w#]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _center_x(span: EvidenceSpan) -> float:
    return (span.bbox[0] + span.bbox[2]) / 2


def _center_y(span: EvidenceSpan) -> float:
    return (span.bbox[1] + span.bbox[3]) / 2


def _in_column(span: EvidenceSpan, column: EvidenceTableColumn) -> bool:
    return column.x_range[0] <= _center_x(span) < column.x_range[1]
