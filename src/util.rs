use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyAny;

pub(crate) fn t_from_any<'a, T>(any: &'a PyAny, to: Option<&str>) -> PyResult<T>
where
    T: FromPyObject<'a>,
{
    let any_type = any.get_type().name();
    let msg = to
        .map(|to| format!("expected '{}' got '{}'", to, any_type))
        .unwrap_or_else(|| format!("invalid argument: '{}'", any_type));
    any.extract::<T>().map_err(|_| type_err(msg))
}

pub(crate) fn type_err(msg: impl Into<String>) -> PyErr {
    exceptions::TypeError::py_err(msg.into())
}

pub(crate) fn val_err(msg: impl Into<String>) -> PyErr {
    exceptions::ValueError::py_err(msg.into())
}

pub(crate) fn io_err(msg: impl Into<String>) -> PyErr {
    exceptions::IOError::py_err(msg.into())
}
