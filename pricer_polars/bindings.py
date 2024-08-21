from polars.plugins import register_plugin_function
from pathlib import Path

import polars as pl

def calculate_black_scholes(*expr):
    expressions = [pl.col(exp) for exp in expr]
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=expressions,
        function_name="calculate_black_scholes",
        is_elementwise=True,
    )
