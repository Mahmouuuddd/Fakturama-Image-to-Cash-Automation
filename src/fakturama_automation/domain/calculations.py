from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from .models import OrderInput, OrderItem, OrderTotals


CENT = Decimal("0.01")
HUNDRED = Decimal("100")


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def line_net(item: OrderItem) -> Decimal:
    discount_factor = Decimal("1") - (item.discount_percent / HUNDRED)
    return money(item.quantity * item.unit_net_price * discount_factor)


def line_vat(item: OrderItem) -> Decimal:
    return money(line_net(item) * item.vat_percent / HUNDRED)


def product_gross_price(item: OrderItem) -> Decimal:
    return money(item.unit_net_price * (Decimal("1") + item.vat_percent / HUNDRED))


def calculated_totals(order: OrderInput) -> OrderTotals:
    total_net = money(sum((line_net(item) for item in order.items), Decimal("0")))
    total_vat = money(sum((line_vat(item) for item in order.items), Decimal("0")))
    return OrderTotals(
        total_net=total_net,
        total_vat=total_vat,
        total_gross=money(total_net + total_vat),
    )


def within_tolerance(left: Decimal, right: Decimal, tolerance: Decimal = CENT) -> bool:
    return abs(left - right) <= tolerance

