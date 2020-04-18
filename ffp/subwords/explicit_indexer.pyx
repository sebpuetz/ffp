# cython: language_level=3
# cython: embedsignature=True
from typing import List, Tuple, Optional, Dict, Iterable

from libc.stdint cimport uint32_t
import cython

cdef class ExplicitIndexer:
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
        if ngram_index is None:
            ngram_index = dict(
                (ngram, idx) for idx, ngram in enumerate(ngrams))
            assert len(ngrams) == len(ngram_index)
            self._bound = len(ngram_index)
        else:
            assert len(set(ngrams)) == len(ngrams)
            self._bound = max(ngram_index.values()) + 1
        self._ngram_index = ngram_index

    @property
    def ngrams(self) -> List[str]:
        """
        Get the list of n-grams in this vocabulary.
        :return: list of n-grams.
        """
        return self._ngrams

    @property
    def ngram_index(self) -> Dict[str, int]:
        """
        Get the ngram-index mapping.
        :return: the ngram-index mapping.
        """
        return self._ngram_index

    @property
    def idx_bound(self) -> int:
        """
        Get the exclusive upper bound of this indexer, i.e. the number of distinct indices.
        :return:
        """
        return self._bound

    @cython.boundscheck(False)
    cpdef subword_indices(self, str word, offset=0, bint bracket=True, bint with_ngrams=False):
        """
        Get the subword indices for the given word.

        :param word: the word
        :param offset: is added to each index
        :param bracket: whether to bracket the word with '<' and '>'
        :param with_ngrams: whether to return the indices with corresponding ngrams
        :return: List of subword indices, optionally as tuples with ngrams
        """
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
