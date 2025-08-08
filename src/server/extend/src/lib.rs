mod calculate;
mod calculate_chars;

use calculate::TextSimilarity;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn similarity_calculate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TextSimilarity>()?;
    Ok(())
}
