"""
Interface to subwords.

Provides access to the subword indexers and a method to extract ngrams from a given string.
"""

from typing import List, Dict, Optional, Iterable

from ffp.vocab_rs import FastTextIndexer, FinalfusionHashIndexer, \
    word_ngrams as _word_ngrams


def word_ngrams(word, min_n=3, max_n=6, bracket=True) -> List[str]:
    """
    Get the ngrams for the given word.
    :param word: word
    :param min_n: lower bound of n-gram range
    :param max_n: upper bound of n-gram range
    :param bracket: whether to bracket the word
    :return: list of n-grams
    """
    return _word_ngrams(word, min_n, max_n, bracket)


def finalfusion_bucketindexer(bucket_exp: 21) -> FinalfusionHashIndexer:
    """
    Returns a new FinalfusionHashIndexer
    :param bucket_exp: pow(2, bucket_exp) defines the number of buckets
    :return: FinalfusionHashIndexer
    """
    if bucket_exp > 64:
        raise ValueError("Bucket exponent needs to be smaller than 64")
    return FinalfusionHashIndexer(bucket_exp)


def fasttext_indexer(n_buckets: 2000000):
    """
    Returns a new fastText hash indexer
    :param n_buckets: defines the number of buckets.
    :return: FinalfusionHashIndexer
    """
    return FastTextIndexer(n_buckets)


class ExplicitIndexer:
    """
    Explicit n-gram indexer.
    """
    def __init__(self,
                 ngrams: List[str],
                 ngram_index: Optional[Dict[str, int]] = None):
        self._ngrams = ngrams
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

    def __call__(self, ngram: str) -> Optional[int]:
        return self.ngram_index.get(ngram)

    def __iter__(self) -> Iterable[str]:
        return iter(self.ngrams)

    def __len__(self) -> int:
        return len(self.ngram_index)
