from polars.plugins import register_plugin_function
from pathlib import Path

from datetime import datetime
import pytz

import polars as pl

def calculate_black_scholes_explicit(*expr):
    expressions = [pl.col(exp) for exp in expr]
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=expressions,
        function_name="calculate_black_scholes_explicit",
        is_elementwise=True,
    )

def calculate_black_scholes(*expr, strike: float, risk_free_rate: float, expiry: datetime):
    expressions = [pl.col(exp) for exp in expr]
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=expressions,
        kwargs={
            "strike": strike,
            "risk_free_rate": risk_free_rate,
            "expiry": pytz.utc.localize(expiry).isoformat(),
        },
        function_name="calculate_black_scholes",
        is_elementwise=True,
    )
