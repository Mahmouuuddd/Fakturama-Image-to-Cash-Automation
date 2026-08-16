from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation


_NUMBER_TOKEN = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:[., \u00a0\u202f]\d{3})+|\d+)"
    r"(?:[.,]\d+)?%?(?![\w])"
)


def parse_decimal_text(value: str) -> Decimal | None:
    """Parse a human-formatted decimal into one canonical numeric value.

    Both common conventions are supported: ``1,234.56`` and ``1.234,56``.
    A lone separator followed by three digits is treated as a grouping mark;
    monetary decimal fractions conventionally use one or two digits.
    """
    source = unicodedata.normalize("NFKC", str(value)).strip()
    parenthesized_negative = source.startswith("(") and source.endswith(")")
    cleaned = re.sub(r"[^0-9,.+\-]", "", source)
    if not cleaned or cleaned.count("+") + cleaned.count("-") > 1:
        return None
    if ("+" in cleaned[1:]) or ("-" in cleaned[1:]):
        return None

    sign = ""
    if cleaned[:1] in {"+", "-"}:
        sign, cleaned = cleaned[0], cleaned[1:]
    if parenthesized_negative:
        if sign:
            return None
        sign = "-"
    if not cleaned:
        return None

    normalized = _normalize_decimal_separators(cleaned)
    if normalized is None or not re.fullmatch(r"\d+(?:\.\d+)?", normalized):
        return None
    try:
        return Decimal(sign + normalized)
    except InvalidOperation:
        return None


def decimal_values_in_text(text: str) -> list[Decimal]:
    """Extract complete numeric tokens and parse them with ``parse_decimal_text``."""
    values: list[Decimal] = []
    normalized = unicodedata.normalize("NFKC", text)
    for match in _NUMBER_TOKEN.finditer(normalized):
        parsed = parse_decimal_text(match.group())
        if parsed is not None:
            values.append(parsed)
    return values


def canonical_decimal(value: str) -> str | None:
    parsed = parse_decimal_text(value)
    return format(parsed, "f") if parsed is not None else None


def _normalize_decimal_separators(value: str) -> str | None:
    if "," in value and "." in value:
        decimal_separator = "," if value.rfind(",") > value.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        grouped, fraction = value.rsplit(decimal_separator, 1)
        if not fraction.isdigit() or not _valid_grouped_integer(
            grouped, grouping_separator
        ):
            return None
        return grouped.replace(grouping_separator, "") + "." + fraction

    separator = "," if "," in value else "." if "." in value else None
    if separator is None:
        return value if value.isdigit() else None

    parts = value.split(separator)
    if any(not part.isdigit() for part in parts):
        return None
    if len(parts) > 2:
        if len(parts[0]) not in {1, 2, 3} or not all(
            len(part) == 3 for part in parts[1:]
        ):
            return None
        return "".join(parts)

    integer, fraction = parts
    if len(fraction) == 3 and 1 <= len(integer) <= 3:
        return integer + fraction
    return integer + "." + fraction


def _valid_grouped_integer(value: str, separator: str) -> bool:
    parts = value.split(separator)
    return (
        1 <= len(parts[0]) <= 3
        and parts[0].isdigit()
        and all(len(part) == 3 and part.isdigit() for part in parts[1:])
    )
