use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Debug, Error, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader};

use finalfusion::compat::fasttext::FastTextIndexer;
use finalfusion::subword::{
    BucketIndexer, FinalfusionHashIndexer, Indexer, NGrams, SubwordIndices,
};
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{wrap_pyfunction, PyObjectProtocol};

mod util;
use util::{io_err, t_from_any, type_err, val_err};

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

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Cutoff {
    MinFreq(MinFreq),
    Target(TargetSize),
}

impl Debug for Cutoff {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Cutoff::MinFreq(i) => write!(f, "{:#?}", i),
            Cutoff::Target(i) => write!(f, "{:#?}", i),
        }
    }
}

/// Defines a cutoff based on target vocabulary size.
#[pyclass]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TargetSize {
    size: usize,
}

#[pymethods]
impl TargetSize {
    #[new]
    fn __new__(obj: &PyRawObject, size: usize) -> PyResult<()> {
        obj.init(TargetSize { size });
        Ok(())
    }

    #[getter]
    fn get_size(self) -> usize {
        self.size
    }

    #[setter]
    fn set_size(&mut self, size: usize) {
        self.size = size;
    }
}

#[pyproto]
impl PyObjectProtocol for TargetSize {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

impl From<TargetSize> for Cutoff {
    fn from(target: TargetSize) -> Self {
        Cutoff::Target(target)
    }
}

/// Defines a cutoff based on minimum frequency.
#[pyclass]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MinFreq {
    freq: usize,
}

#[pymethods]
impl MinFreq {
    #[new]
    fn __new__(obj: &PyRawObject, freq: usize) -> PyResult<()> {
        obj.init(MinFreq { freq });
        Ok(())
    }

    #[getter]
    fn get_freq(self) -> usize {
        self.freq
    }

    #[setter]
    fn set_freq(&mut self, freq: usize) {
        self.freq = freq;
    }
}

#[pyproto]
impl PyObjectProtocol for MinFreq {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

impl From<MinFreq> for Cutoff {
    fn from(freq: MinFreq) -> Self {
        Cutoff::MinFreq(freq)
    }
}

impl<'a> FromPyObject<'a> for Cutoff {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(&target) = ob.downcast_ref::<TargetSize>() {
            Ok(target.into())
        } else if let Ok(&freq) = ob.downcast_ref::<MinFreq>() {
            Ok(freq.into())
        } else {
            type_err(format!(
                "Expected MinFreq or TargetSize, not '{}'",
                ob.get_type().name()
            ))
            .into()
        }
    }
}

impl FromPy<Cutoff> for PyObject {
    fn from_py(cutoff: Cutoff, py: Python) -> Self {
        match cutoff {
            Cutoff::MinFreq(freq) => freq.into_py(py),
            Cutoff::Target(size) => size.into_py(py),
        }
    }
}

#[pyfunction]
fn count_and_sort_tokens(corpus: &PyAny, cutoff: Cutoff) -> PyResult<PyObject> {
    let corpus = t_from_any::<&str>(corpus, Some("string"))?;
    let file = File::open(corpus).map_err(|e| {
        io_err(format!(
            "Could not open the corpus for reading: {}\n{}",
            corpus, e
        ))
    })?;
    let bytes = file
        .metadata()
        .map_err(|e| io_err(format!("Could not get file metadata\n{}", e)))?
        .len();
    let pb = ProgressBar::new(bytes);
    pb.enable_steady_tick(1000);
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{elapsed_precise} {bytes}/{total_bytes} ETA: {eta_precise}"),
    );
    let read = BufReader::new(pb.wrap_read(file));
    let token_counts = count_from_reader(read)?;
    let gil = Python::acquire_gil();
    let res = filter_vocab_items(token_counts, cutoff).into_py(gil.python());
    pb.finish();
    Ok(res)
}

#[pyfunction]
fn count_and_sort_tokens_and_ngrams(
    corpus: &PyAny,
    token_cutoff: Cutoff,
    ngram_cutoff: Cutoff,
    min_n: usize,
    max_n: usize,
) -> PyResult<PyObject> {
    let corpus = t_from_any::<&str>(corpus, Some("string"))?;
    let file = File::open(corpus).map_err(|e| {
        io_err(format!(
            "Could not open the corpus for reading: {}\n{}",
            corpus, e
        ))
    })?;
    let bytes = file
        .metadata()
        .map_err(|e| io_err(format!("Could not get file metadata\n{}", e)))?
        .len();
    let pb = ProgressBar::new(bytes);
    pb.enable_steady_tick(1000);
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{elapsed_precise} {bytes}/{total_bytes} ETA: {eta_precise}"),
    );
    let read = BufReader::new(pb.wrap_read(file));
    let token_counts = count_from_reader(read)?;
    let mut ngram_counts = HashMap::new();
    for (token, count) in token_counts.iter() {
        let token = format!("<{}>", token);
        for ngram in NGrams::new(&token, min_n, max_n) {
            let cnt = ngram_counts.entry(ngram.to_string()).or_default();
            *cnt += *count;
        }
    }
    let token_res = filter_vocab_items(token_counts, token_cutoff);
    let ngram_res = filter_vocab_items(ngram_counts, ngram_cutoff);
    let gil = Python::acquire_gil();
    let res = (token_res, ngram_res).into_py(gil.python());
    pb.finish();
    Ok(res)
}

fn count_from_reader<R>(read: R) -> PyResult<HashMap<String, usize>>
where
    R: BufRead,
{
    let mut counts = HashMap::new();
    for line in read.lines() {
        let line = line.map_err(|e| io_err(format!("Failed to read from corpus: {}", e)))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        for part in line.split_whitespace() {
            let cnt = counts.entry(part.to_owned()).or_default();
            *cnt += 1;
        }
    }
    Ok(counts)
}

fn filter_vocab_items(counts: HashMap<String, usize>, cutoff: Cutoff) -> (Vec<String>, Vec<usize>) {
    let items_with_counts = match cutoff {
        Cutoff::MinFreq(min) => {
            counted_into_sorted(counts.into_iter().filter(|(_, count)| *count >= min.freq))
        }
        Cutoff::Target(target) => {
            if target.size == 0 {
                Vec::new()
            } else {
                // the target_idx is one off the target size since indexing starts at 0
                let mut target_idx = target.size - 1;
                let mut items = counted_into_sorted(counts.into_iter());
                if let (Some((_, cnt_at_target)), Some((_, cnt_after_target))) =
                    (items.get(target_idx), items.get(target_idx + 1))
                {
                    if cnt_at_target == cnt_after_target {
                        while target_idx > 0 && items[target_idx].1 == *cnt_at_target {
                            target_idx -= 1;
                        }
                    }
                }
                items.truncate(target_idx + 1);
                items
            }
        }
    };
    let mut items = Vec::with_capacity(items_with_counts.len());
    let mut counts = Vec::with_capacity(items_with_counts.len());
    for (item, count) in items_with_counts {
        items.push(item);
        counts.push(count);
    }
    (items, counts)
}

fn counted_into_sorted(iter: impl Iterator<Item = (String, usize)>) -> Vec<(String, usize)> {
    let mut items = iter.collect::<Vec<_>>();
    items.sort_unstable_by(|(t1, c1), (t2, c2)| match c2.cmp(c1) {
        Ordering::Equal => t1.cmp(t2),
        o => o,
    });
    items
}

#[pymodule]
fn vocab_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFinalfusionHashIndexer>()?;
    m.add_class::<PyFastTextIndexer>()?;
    m.add_class::<MinFreq>()?;
    m.add_class::<TargetSize>()?;
    m.add_wrapped(wrap_pyfunction!(word_ngrams))?;
    m.add_wrapped(wrap_pyfunction!(count_and_sort_tokens_and_ngrams))?;
    m.add_wrapped(wrap_pyfunction!(count_and_sort_tokens))
}
