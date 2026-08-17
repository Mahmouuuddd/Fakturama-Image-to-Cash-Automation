from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import TypeVar

from .models import Address, Debtor, DebtorCandidate, ProductCandidate


T = TypeVar("T")


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def normalize_postal_code(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", normalize_text(value))


def addresses_match(expected: Address, actual: Address) -> bool:
    return all(
        (
            normalize_text(expected.street) == normalize_text(actual.street),
            normalize_postal_code(expected.zip) == normalize_postal_code(actual.zip),
            normalize_text(expected.city) == normalize_text(actual.city),
            normalize_text(expected.country) == normalize_text(actual.country),
        )
    )


def has_distinct_delivery_address(debtor: Debtor) -> bool:
    return debtor.delivery_address is not None and not addresses_match(
        debtor.billing_address, debtor.delivery_address
    )


def main_address_only(debtor: Debtor) -> Debtor:
    """Represent a Fakturama Debtor when no second address is created."""
    if not has_distinct_delivery_address(debtor):
        return debtor
    return debtor.model_copy(update={"delivery_address": None})


def debtor_matches(expected: Debtor, actual: DebtorCandidate) -> bool:
    company_match = (
        not expected.company
        or not actual.company
        or normalize_text(expected.company) == normalize_text(actual.company)
    )
    first_match = (
        not expected.first_name
        or not actual.first_name
        or normalize_text(expected.first_name) == normalize_text(actual.first_name)
    )
    last_match = (
        not expected.last_name
        or not actual.last_name
        or normalize_text(expected.last_name) == normalize_text(actual.last_name)
    )
    identity_match = (
        (expected.company and normalize_text(expected.company) == normalize_text(actual.company))
        or (
            bool(expected.first_name or expected.last_name)
            and normalize_text(expected.first_name) == normalize_text(actual.first_name)
            and normalize_text(expected.last_name) == normalize_text(actual.last_name)
        )
    )
    return all(
        (
            identity_match,
            company_match,
            first_match,
            last_match,
            normalize_postal_code(expected.billing_address.zip)
            == normalize_postal_code(actual.billing_address.zip),
            normalize_text(expected.billing_address.city)
            == normalize_text(actual.billing_address.city),
        )
    )


def product_matches(sku: str, actual: ProductCandidate) -> bool:
    return normalize_text(sku) == normalize_text(actual.sku)


def exact_matches(
    candidates: Sequence[T], predicate: Callable[[T], bool]
) -> list[T]:
    return [candidate for candidate in candidates if predicate(candidate)]
