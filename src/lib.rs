use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use pricer::{option::get_call, BlackScholes};

use chrono::prelude::Utc;
use chrono::DateTime;

#[polars_expr(output_type=Float64)]
fn calculate_black_scholes(inputs: &[Series]) -> PolarsResult<Series> {
    let strike = inputs[0].f64()?;
    let volatility = inputs[1].f64()?;
    let underlying = inputs[2].f64()?;
    let rfr = inputs[3].f64()?;
    let expiry = inputs[4].datetime()?;
    let out: ChunkedArray<Float64Type> = strike
        .into_iter()
        .zip(volatility.into_iter())
        .zip(underlying.into_iter())
        .zip(rfr.into_iter())
        .zip(expiry.into_iter())
        .map(
            |((((strike, volatility), underlying), rfr), expiry)| -> Option<f64> {
                let strike = strike?;
                let volatility = volatility?;
                let underlying = underlying?;
                let rfr = rfr?;
                let expiry = DateTime::from_timestamp_micros(expiry?)?;
                let call = get_call(strike, expiry);
                call.value_black_scholes(Utc::now(), volatility, underlying, rfr)
                    .ok()
            },
        )
        .collect();
    Ok(out.with_name("black_scholes").into_series())
}
