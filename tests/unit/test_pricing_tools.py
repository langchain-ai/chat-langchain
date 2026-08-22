"""Tests for pricing extraction validation."""

from src.tools.pricing_tools import _looks_like_pricing


def test_looks_like_pricing_rejects_navigation_chrome():
    navigation = "Products LangSmith Platform Observability Evaluation Company Developer Plus Enterprise per seat /mo LCU LSU"

    assert not _looks_like_pricing(navigation)


def test_looks_like_pricing_accepts_currency_amount():
    assert _looks_like_pricing("$39 per seat per month")
