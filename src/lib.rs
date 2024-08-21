use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use pricer::{option::get_call, BlackScholes};

use chrono::prelude::Utc;
use chrono::DateTime;

use serde::Deserialize;

#[derive(Deserialize)]
pub struct BlackScholesKwargs {
    strike: f64,
    risk_free_rate: f64,
    expiry: String,
}

fn parse_python_date(datetime_str: &str) -> PolarsResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(datetime_str)
        .map(|dt| dt.into())
        .map_err(|_| PolarsError::SchemaMismatch("Failed to parse python date".into()))
}

#[polars_expr(output_type=Float64)]
fn calculate_black_scholes(inputs: &[Series], kwargs: BlackScholesKwargs) -> PolarsResult<Series> {
    let underlying = inputs[0].f64()?;
    let volatility = inputs[1].f64()?;
    let expiry = parse_python_date(&kwargs.expiry)?;
    let call = get_call(kwargs.strike, expiry);
    let now = Utc::now();
    let out: Float64Chunked =
        arity::binary_elementwise_values(underlying, volatility, |ul, vl| -> f64 {
            call.value_black_scholes(now, vl, ul, kwargs.risk_free_rate)
                .unwrap_or_default()
        });
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn calculate_black_scholes_explicit(inputs: &[Series]) -> PolarsResult<Series> {
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
