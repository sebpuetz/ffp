use finalfusion::compat::fasttext::FastTextIndexer;
use finalfusion::subword::{
    BucketIndexer, FinalfusionHashIndexer, Indexer, NGrams, SubwordIndices,
};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{wrap_pyfunction, PyObjectProtocol};

mod util;
use util::{t_from_any, val_err};

/// FinalFusionHashIndexer(,/,buckets_exp)
/// --
///
/// `buckets_exp` defines the number of available hashing buckets as
/// `n_buckets = pow(2, buckets_exp)`
///
/// Default hash-based indexer in finalfusion.
///
/// May assign the same index to different inputs.
#[pyclass(name=FinalfusionHashIndexer)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct PyFinalfusionHashIndexer {
    pub indexer: FinalfusionHashIndexer,
}

#[pymethods]
impl PyFinalfusionHashIndexer {
    #[args(buckets_exp = 21)]
    #[new]
    fn __new__(obj: &PyRawObject, buckets_exp: usize) -> PyResult<()> {
        if buckets_exp > 64 {
            return val_err("buckets_exp cannot be larger than 64").into();
        }
        let indexer = FinalfusionHashIndexer::new(buckets_exp);
        obj.init(PyFinalfusionHashIndexer { indexer });
        Ok(())
    }

    #[getter]
    fn idx_bound(self) -> u64 {
        self.indexer.upper_bound()
    }

    #[getter]
    fn get_buckets_exp(&self) -> usize {
        self.indexer.buckets()
    }

    #[call]
    fn call(self, s: &PyAny) -> PyResult<u64> {
        let s = t_from_any::<&str>(s, Some("string"))?;
        Ok(self
            .indexer
            .index_ngram(&s.into())
            .expect("FinalfusionHashIndexer should index all inputs."))
    }

    #[args(min_n = 3, max_n = 6, bracket = "true", offest = 0)]
    fn subword_indices(
        self,
        s: &PyAny,
        min_n: usize,
        max_n: usize,
        bracket: bool,
        offset: u64,
    ) -> PyResult<Vec<u64>> {
        if min_n > max_n || min_n == 0 {
            return val_err("min_n must be smaller than max_n and nonzero").into();
        }
        let string = t_from_any::<&str>(s, Some("string"))?;
        Ok(subword_indices_impl(
            self.indexer,
            string,
            min_n,
            max_n,
            bracket,
            offset,
        ))
    }
}

#[pyproto]
impl PyObjectProtocol for PyFinalfusionHashIndexer {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FinalfusionHashIndexer(n_buckets: {})",
            self.indexer.upper_bound()
        ))
    }
}

/// FastTextIndexer(,/,buckets_exp)
/// --
///
/// `n_buckets` defines the number of available hashing buckets.
///
/// Default hash-based indexer in finalfusion.
///
/// May assign the same index to different inputs.
#[pyclass(name=FastTextIndexer)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct PyFastTextIndexer {
    pub indexer: FastTextIndexer,
}

#[pymethods]
impl PyFastTextIndexer {
    #[args(n_buckets = 2000000)]
    #[new]
    fn __new__(obj: &PyRawObject, n_buckets: usize) -> PyResult<()> {
        let indexer = FastTextIndexer::new(n_buckets);
        obj.init(PyFastTextIndexer { indexer });
        Ok(())
    }

    #[getter]
    fn idx_bound(self) -> u64 {
        self.indexer.upper_bound()
    }

    #[call]
    fn call(self, ngram: &PyAny) -> PyResult<u64> {
        let s = t_from_any::<&str>(ngram, Some("string"))?;
        Ok(self
            .indexer
            .index_ngram(&s.into())
            .expect("FastTextIndexer should index all inputs."))
    }

    #[args(min_n = 3, max_n = 6, bracket = "true", offset = 0)]
    fn subword_indices(
        self,
        s: &PyAny,
        min_n: usize,
        max_n: usize,
        bracket: bool,
        offset: u64,
    ) -> PyResult<Vec<u64>> {
        if min_n > max_n || min_n == 0 {
            return val_err("min_n must be smaller than max_n and nonzero").into();
        }
        let string = t_from_any::<&str>(s, Some("string"))?;
        Ok(subword_indices_impl(
            self.indexer,
            string,
            min_n,
            max_n,
            bracket,
            offset,
        ))
    }
}

#[pyproto]
impl PyObjectProtocol for PyFastTextIndexer {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FastTextIndexer(n_buckets: {})",
            self.indexer.upper_bound()
        ))
    }
}

fn subword_indices_impl<I: Indexer>(
    indexer: I,
    string: &str,
    min_n: usize,
    max_n: usize,
    bracket: bool,
    offset: u64,
) -> Vec<u64> {
    if bracket {
        let mut s = String::with_capacity(string.len() + 2);
        s.push('<');
        s.push_str(string);
        s.push('>');
        s.subword_indices(min_n, max_n, &indexer)
            .map(|i| i + offset)
            .collect()
    } else {
        string
            .subword_indices(min_n, max_n, &indexer)
            .map(|i| i + offset)
            .collect()
    }
}

#[pyfunction]
fn word_ngrams(string: &str, min_n: usize, max_n: usize, bracket: bool) -> PyResult<Vec<PyObject>> {
    if min_n > max_n || min_n == 0 {
        return val_err("min_n must be smaller than max_n and nonzero").into();
    }
    let gil = Python::acquire_gil();
    Ok(if bracket {
        let mut s = String::with_capacity(string.len() + 2);
        s.push('<');
        s.push_str(string);
        s.push('>');
        NGrams::new(&s, min_n, max_n)
            .map(|s| s.as_str().into_py(gil.python()))
            .collect()
    } else {
        NGrams::new(&string, min_n, max_n)
            .map(|s| s.as_str().into_py(gil.python()))
            .collect()
    })
}

#[pymodule]
fn vocab_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFinalfusionHashIndexer>()?;
    m.add_class::<PyFastTextIndexer>()?;
    m.add_wrapped(wrap_pyfunction!(word_ngrams))
}
