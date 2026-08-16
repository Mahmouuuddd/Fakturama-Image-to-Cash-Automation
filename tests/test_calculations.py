import json
from decimal import Decimal
from pathlib import Path

from fakturama_automation.domain.calculations import (
    calculated_totals,
    line_net,
    product_gross_price,
)
from fakturama_automation.domain.models import OrderInput


def load_example() -> OrderInput:
    data = json.loads(Path("examples/order.json").read_text(encoding="utf-8"))
    return OrderInput.model_validate(data)


def test_financial_calculations_use_expected_rounding() -> None:
    order = load_example()

    assert line_net(order.items[0]) == Decimal("180.00")
    assert product_gross_price(order.items[0]) == Decimal("119.00")
    assert calculated_totals(order).total_net == Decimal("230.00")
    assert calculated_totals(order).total_vat == Decimal("37.70")
    assert calculated_totals(order).total_gross == Decimal("267.70")

