# cython: language_level=3
# cython: embedsignature=True
# cython: infer_types=True
from typing import List, Tuple, Optional, Dict, Iterable, Union

from libc.stdint cimport uint32_t
import cython

cdef class ExplicitIndexer:
    """
    ExplicitIndexer

    Explicit Indexers do not index n-grams through hashing but define an actual lookup table.

    It can be constructed from a list of **unique** ngrams. In that case, the ith ngram in the
    list will be mapped to index i. It is also possible to pass a mapping via `ngram_index`
    which allows mapping multiple ngrams to the same value.

    N-grams can be indexed directly through the `__call__` method or all n-grams in a string
    can be indexed in bulk through the `subword_indices` method.

    `subword_indices` optionally returns tuples of form `(ngram, idx)`, otherwise a list of
    indices belonging to the input string is returned.
    """
    cdef dict _ngram_index
    cdef list _ngrams
    cdef Py_ssize_t _bound
    cdef public uint32_t min_n
    cdef public uint32_t max_n

    def __init__(self,
                 ngrams: List[str],
                 ngram_range: Tuple[int, int] = (3, 6),
                 ngram_index: Optional[Dict[str, int]] = None):
        self._ngrams = ngrams
        self.min_n, self.max_n = ngram_range
        assert ngram_range[1] >= ngram_range[0] > 0, \
            f"Min_n{ngram_range[0]} must be greater than 0, max_n{ngram_range[1]} must be >= min_n"
        if ngram_index is None:
            ngram_index = dict(
                (ngram, idx) for idx, ngram in enumerate(ngrams))
            assert len(ngrams) == len(ngram_index)
            self._bound = len(ngram_index)
        else:
            assert len(set(ngrams)) == len(ngrams)
            max_idx = max(ngram_index.values())
            n_unique_vals = len(set(ngram_index.values()))
            assert max_idx + 1 == n_unique_vals, \
                f"Max idx ({max_idx}) is required to be n_values - 1 ({n_unique_vals})"
            self._bound = max_idx + 1
        self._ngram_index = ngram_index

    @property
    def ngrams(self) -> List[str]:
        """
        Get the list of n-grams.

        Returns
        -------
        ngrams : list
            The list of in-vocabulary n-grams.
        """
        return self._ngrams

    @property
    def ngram_index(self) -> Dict[str, int]:
        """
        Get the ngram-index mapping.

        Returns
        -------
        ngram_index : dict
            The ngram -> index mapping.
        """
        return self._ngram_index

    @property
    def idx_bound(self) -> int:
        """
        Get the **exclusive** upper bound

        This is the number of distinct indices.

        Returns
        -------
        idx_bound : int
            Exclusive upper bound of the indexer.
        """
        return self._bound

    cpdef subword_indices(self, str word, offset=0, bint bracket=True, bint with_ngrams=False):
        """
        Get the subword indices for a word.
        
        Parameters
        ----------
        word : str
            The string to extract n-grams from
        offset : int
            The offset to add to the index, e.g. the length of the word-vocabulary.
        bracket : bool
            Toggles bracketing the input string with `<` and `>`
        with_ngrams : bool
            Toggles returning tuples of (ngram, idx)
        
        Returns
        -------
        indices : list
            List of n-gram indices, optionally as `(str, int)` tuples.
        
        Raises
        ------
        TypeError
            If `word` is None.
        """
        if word is None:
            raise TypeError("Can't extract ngrams for None type")
        if bracket:
            word = "<%s>" % word
        cdef Py_ssize_t i, j
        cdef Py_ssize_t length = len(word)
        cdef list ngrams = []
        if length < self.min_n:
            return ngrams
        cdef Py_ssize_t max_n = min(self.max_n, length)
        for i in range(length + 1 - self.min_n):
            for j in range(max_n, self.min_n-1, -1):
                if j + i <= length:
                    ngram = word[i:i + j]
                    idx = self._ngram_index.get(ngram)
                    if idx is None:
                        continue
                    if with_ngrams:
                        ngrams.append((ngram, idx + offset))
                    else:
                        ngrams.append(idx + offset)
        return ngrams

    def __call__(self, ngram: str) -> Optional[int]:
        return self.ngram_index.get(ngram)

    def __iter__(self) -> Iterable[str]:
        return iter(self._ngrams)

    def __len__(self) -> int:
        return len(self._ngram_index)
