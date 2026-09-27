"""Currency tools, registered in the demo by import path.

Stands in for a module that is expensive to import (a vendor SDK, a client that opens
connections...): the demo shows it is only imported once one of its tools is loaded.
"""

_RATES_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "INR": 0.012}


def get_exchange_rate(base: str, quote: str) -> str:
    """Get today's exchange rate between two currencies.

    Args:
        base: ISO code of the currency to convert from, e.g. "USD".
        quote: ISO code of the currency to convert to, e.g. "EUR".
    """
    return f"1 {base} = {_RATES_TO_USD[base] / _RATES_TO_USD[quote]:.4f} {quote}"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money from one currency to another at today's rate.

    Args:
        amount: Amount to convert.
        from_currency: ISO code of the source currency, e.g. "USD".
        to_currency: ISO code of the target currency, e.g. "EUR".
    """
    converted = amount * _RATES_TO_USD[from_currency] / _RATES_TO_USD[to_currency]
    return f"{amount:g} {from_currency} = {converted:.2f} {to_currency}"
