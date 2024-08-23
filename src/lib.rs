use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use pricer::{option::get_call, BlackScholes};

use chrono::prelude::Utc;
use chrono::DateTime;

use itertools::multizip;
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

fn bind_optionals_to_black_scholes(
    (strike, volatility, underlying, risk_free_rate, expiry): (
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<i64>,
    ),
) -> Option<f64> {
    let strike = strike?;
    let volatility = volatility?;
    let underlying = underlying?;
    let risk_free_rate = risk_free_rate?;
    let expiry = expiry.and_then(|e| DateTime::from_timestamp_micros(e))?;
    let call = get_call(strike, expiry);
    call.value_black_scholes(Utc::now(), volatility, underlying, risk_free_rate)
        .ok()
}

#[polars_expr(output_type=Float64)]
fn calculate_black_scholes_explicit(inputs: &[Series]) -> PolarsResult<Series> {
    let bound_tuple = (
        inputs[0].f64()?.into_iter(),
        inputs[1].f64()?.into_iter(),
        inputs[2].f64()?.into_iter(),
        inputs[3].f64()?.into_iter(),
        inputs[4].datetime()?.into_iter(),
    );
    let out: ChunkedArray<Float64Type> = multizip(bound_tuple)
        .map(bind_optionals_to_black_scholes)
        .collect();
    Ok(out.with_name("black_scholes").into_series())
}
