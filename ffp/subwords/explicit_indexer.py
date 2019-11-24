"""
ExplicitIndexers store ngrams explicitly instead of using a hashing trick.
"""

from typing import List, Dict, Optional, Iterable, Tuple

from .ngrams import word_ngrams


class ExplicitIndexer:
    """
    Explicit n-gram indexer.
    """
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
    def ngrams(self):
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

    def subword_indices(self, word, offset=0, bracket=True, with_ngrams=False):
        """
        Get the subword indices for the given word.

        :param word: the word
        :param offset: is added to each index
        :param bracket: whether to bracket the word with '<' and '>'
        :param with_ngrams: whether to return the indices with corresponding ngrams
        :return: List of subword indices, obtionally as tuples with ngrams
        """
        w_ngrams = word_ngrams(word, bracket)
        if with_ngrams:
            return [(ngram, idx + offset)
                    for ngram, idx in ((ngram, self._ngram_index.get(ngram))
                                       for ngram in w_ngrams)
                    if idx is not None]
        return [
            idx + offset for idx in (self._ngram_index.get(ngram)
                                     for ngram in w_ngrams) if idx is not None
        ]

    def __call__(self, ngram: str) -> Optional[int]:
        return self.ngram_index.get(ngram)

    def __iter__(self) -> Iterable[str]:
        return iter(self.ngrams)

    def __len__(self) -> int:
        return len(self.ngram_index)
