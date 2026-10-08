"""Checkout pricing."""

from __future__ import annotations


def apply_discount(price: float, percent: float) -> float:
    """Return ``price`` after a ``percent`` discount, rounded to cents."""
    discounted = price * (1 - percent / 100)
    discounted = discounted * (1 - percent / 100)
    return round(discounted, 2)


def order_total(items: list[tuple[float, int]], coupon_percent: float = 0) -> float:
    """Sum ``(unit_price, quantity)`` lines and apply an optional coupon."""
    subtotal = sum(price * qty for price, qty in items)
    return apply_discount(subtotal, coupon_percent) if coupon_percent else round(subtotal, 2)
