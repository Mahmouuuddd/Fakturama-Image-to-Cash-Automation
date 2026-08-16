import json
from pathlib import Path

from fakturama_automation.domain.matching import debtor_matches, normalize_text
from fakturama_automation.domain.models import DebtorCandidate, OrderInput


def test_text_matching_is_case_and_whitespace_insensitive() -> None:
    assert normalize_text("  Example   GmbH ") == normalize_text("example gmbh")


def test_debtor_match_requires_all_identity_fields() -> None:
    order = OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )
    expected = order.debtor
    candidate = DebtorCandidate(
        record_id="debtor-1",
        company="example manufacturing gmbh",
        first_name="Ada",
        last_name="Lovelace",
        billing_address=expected.billing_address,
    )

    assert debtor_matches(expected, candidate)
    assert not debtor_matches(
        expected,
        candidate.model_copy(
            update={
                "billing_address": candidate.billing_address.model_copy(
                    update={"city": "Hamburg"}
                )
            }
        ),
    )

