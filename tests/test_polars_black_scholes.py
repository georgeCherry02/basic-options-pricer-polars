from datetime import datetime
from dateutil.relativedelta import relativedelta

from pricer_polars.bindings import calculate_black_scholes

import polars as pl

def in_four_months():
    return datetime.now() + relativedelta(days=121)

def is_close(value: float, expected: float, precision: float) -> bool:
    return abs(expected - value) < precision


def test_polars_integration():
    df = pl.DataFrame({
        "strike": [40.0, 45.0, 50.0],
        "volatility": [0.4] * 3,
        "underlying": [40.0] * 3,
        "rfr": [0.04] * 3,
        "expiry": [in_four_months()] * 3,
    })
    out = df.with_columns(
        black_scholes = calculate_black_scholes("strike", "volatility", "underlying", "rfr", "expiry")
    )
    value = out.row(1, named=True)["black_scholes"]
    assert is_close(value, 2.0557, 0.0001), "Valued 4 month call correctly"

