"""
Finalfusion Subword Vocabularies
"""

import collections
import struct
from abc import abstractmethod
from os import PathLike
from typing import List, Optional, Tuple, Any, Union, Dict, BinaryIO

from ffp.io import ChunkIdentifier, find_chunk, _write_binary, _read_binary, Chunk
from ffp.subwords import ExplicitIndexer, FastTextIndexer, FinalfusionHashIndexer, word_ngrams
from ffp.vocab.vocab import Vocab, _validate_words_and_create_index, _calculate_serialized_size, \
    _write_words_binary, _read_items
from ffp.vocab.cutoff import Cutoff, _count_words, _filter_and_sort


class SubwordVocab(Vocab):
    """
    Interface for vocabularies with subword lookups.
    """
    def idx(self, item: str, default=None) -> Optional[Union[List[int], int]]:
        idx = self.word_index.get(item)
        if idx is not None:
            return idx
        subwords = self.subword_indices(item)
        if subwords:
            return subwords
        return default

    @property
    def idx_bound(self) -> int:
        return len(self) + self.subword_indexer.idx_bound

    @property
    def min_n(self) -> int:
        """
        Get the lower bound of the range of extracted n-grams.

        Returns
        -------
        min_n : int
            lower bound of n-gram range.
        """
        return self.subword_indexer.min_n

    @property
    def max_n(self) -> int:
        """
        Get the upper bound of the range of extracted n-grams.

        Returns
        -------
        max_n : int
            upper bound of n-gram range.
        """
        return self.subword_indexer.max_n

    @property
    @abstractmethod
    def subword_indexer(
            self
    ) -> Union[ExplicitIndexer, FinalfusionHashIndexer, FastTextIndexer]:
        """
        Get this vocab's subword Indexer.

        The subword indexer produces indices for n-grams.

        In case of bucket vocabularies, this is a hash-based indexer
        (:class:`.FinalfusionHashIndexer`, :class:`.FastTextIndexer`). For explicit subword
        vocabularies, this is an :class:`.ExplicitIndexer`.

        Returns
        -------
        subword_indexer : ExplicitIndexer, FinalfusionHashIndexer, FastTextIndexer
            The subword indexer of the vocabulary.
        """
    def subwords(self, item: str, bracket: bool = True) -> List[str]:
        """
        Get the n-grams of the given item as a list.

        The n-gram range is determined by the `min_n` and `max_n` values.

        Parameters
        ----------
        item : str
            The query item to extract n-grams from.
        bracket : bool
            Toggles bracketing the item with '<' and '>' before extraction.

        Returns
        -------
        ngrams : List[str]
            List of n-grams.
        """
        return word_ngrams(item, self.min_n, self.max_n, bracket)

    def subword_indices(self, item: str, bracket: bool = True) -> List[int]:
        """
        Get the subword indices for the given item.

        This list does not contain the index for known items.

        Parameters
        ----------
        item : str
            The query item.
        bracket : bool
            Toggles bracketing the item with '<' and '>' before extraction.

        Returns
        -------
        indices : List[int]
            The list of subword indices.
        """
        return self.subword_indexer.subword_indices(item,
                                                    offset=len(self.words),
                                                    bracket=bracket)

    def __getitem__(self, item: str) -> Union[int, List[int]]:
        idx = self.word_index.get(item)
        if idx is not None:
            return idx
        subwords = self.subword_indices(item)
        if subwords:
            return subwords
        raise KeyError(f"No indices found for {item}")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.min_n != other.min_n:
            return False
        if self.max_n != other.max_n:
            return False
        return super(SubwordVocab, self).__eq__(other)


class FinalfusionBucketVocab(Chunk, SubwordVocab):
    """
    Finalfusion Bucket Vocabulary.
    """
    def __init__(self,
                 words: List[str],
                 indexer: FinalfusionHashIndexer = None,
                 index: Optional[Dict[str, int]] = None):
        """
        Initialize a FinalfusionBucketVocab.

        Initializes the vocabulary with the given words and optional index and
        indexer.

        If no indexer is passed, a FinalfusionHashIndexer with bucket exponent
        21 is used.

        If no index is given, the nth word in the `words` list is assigned
        index `n`. The word list cannot contain duplicate entries and it needs
        to be of same length as the index.

        Parameters
        ----------
        words : List[str]
            List of unique words
        indexer : FinalfusionHashIndexer, optional
            Subword indexer to use for the vocabulary. Defaults to an indexer
            with 2^21 buckets with range 3-6.
        index : Dict[str, int], optional
            Dictionary providing an entry -> index mapping.

        Raises
        ------
        ValueError
            if the length of `index` and `word` doesn't match.
        AssertionError
            If the indexer is not a FinalfusionHashIndexer.
        """
        if indexer is None:
            indexer = FinalfusionHashIndexer(21)
        assert isinstance(indexer, FinalfusionHashIndexer)
        super().__init__()
        self._index = _validate_words_and_create_index(words, index)
        self._words = words
        self._indexer = indexer

    @staticmethod
    def from_corpus(
            file: Union[str, bytes, int, PathLike],
            cutoff: Optional[Cutoff] = None,
            indexer: Optional[FinalfusionHashIndexer] = None,
    ) -> Tuple['FinalfusionBucketVocab', List[int]]:
        """
        Build a Finalfusion Bucket Vocabulary from a corpus.

        Parameters
        ----------
        file : str, bytes, int, PathLike
            File with white-space separated tokens.
        cutoff : Cutoff
            Frequency cutoff or target size to restrict vocabulary size. Defaults to
            minimum frequency cutoff of 30.
        indexer : FinalfusionHashIndexer
            Subword indexer to use for the vocabulary. Defaults to an indexer
            with 2^21 buckets with range 3-6.

        Returns
        -------
        (vocab, counts) : Tuple[FinalfusionBucketVocab, List[int]]
            Tuple containing the Vocabulary as first item and counts of in-vocabulary items
            as the second item.

        Raises
        ------
        AssertionError
            If the indexer is not a FinalfusionHashIndexer.
        """
        assert indexer is None or isinstance(indexer, FinalfusionHashIndexer)
        cnt = _count_words(file)
        if cutoff is None:
            cutoff = Cutoff(30, mode='min_freq')
        words, counts = _filter_and_sort(cnt, cutoff)
        return FinalfusionBucketVocab(words, indexer), counts

    def to_explicit(self) -> 'ExplicitVocab':
        """
        Returns a Vocabulary with explicit storage built from this vocab.

        Returns
        -------
        explicit_vocab : ExplicitVocab
            The converted vocabulary.
        """
        return _to_explicit(self)

    def write_chunk(self, file: BinaryIO):
        _write_bucket_vocab(self, file)

    @property
    def subword_indexer(self) -> FinalfusionHashIndexer:
        return self._indexer

    @property
    def words(self) -> list:
        return self._words

    @property
    def word_index(self) -> dict:
        return self._index

    @staticmethod
    def read_chunk(file: BinaryIO) -> 'FinalfusionBucketVocab':
        length, min_n, max_n, buckets = _read_binary(file, "<QIII")
        words, index = _read_items(file, length)
        indexer = FinalfusionHashIndexer(buckets, min_n, max_n)
        return FinalfusionBucketVocab(words, indexer, index)

    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.BucketSubwordVocab

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not isinstance(other.subword_indexer, type(self.subword_indexer)):
            return False
        if self.subword_indexer.idx_bound != other.subword_indexer.idx_bound:
            return False
        return super(FinalfusionBucketVocab, self).__eq__(other)


class FastTextVocab(Chunk, SubwordVocab):
    """
    FastText vocabulary
    """
    def __init__(self,
                 words: List[str],
                 indexer: FastTextIndexer = None,
                 index: Optional[Dict[str, int]] = None):
        """
        Initialize a FastTextVocab.

        Initializes the vocabulary with the given words and optional index and
        indexer.

        If no indexer is passed, a FastTextIndexer with 2,000,000 buckets is
        used.

        If no index is given, the nth word in the `words` list is assigned
        index `n`. The word list cannot contain duplicate entries and it needs
        to be of same length as the index.

        Parameters
        ----------
        words : List[str]
            List of unique words
        indexer : FastTextIndexer, optional
            Subword indexer to use for the vocabulary. Defaults to an indexer
            with 2,000,000 buckets with range 3-6.
        index : Dict[str, int], optional
            Dictionary providing an entry -> index mapping.

        Raises
        ------
        ValueError
            if the length of `index` and `word` doesn't match.
        AssertionError
            If the indexer is not a FastTextIndexer.
        """
        if indexer is None:
            indexer = FastTextIndexer(2000000)
        assert isinstance(indexer, FastTextIndexer)
        super().__init__()
        self._index = _validate_words_and_create_index(words, index)
        self._words = words
        self._indexer = indexer

    @staticmethod
    def from_corpus(
            file: Union[str, bytes, int, PathLike],
            cutoff: Optional[Cutoff] = None,
            indexer: Optional[FastTextIndexer] = None,
    ) -> Tuple['FastTextVocab', List[int]]:
        """
        Build a fastText vocabulary from a corpus.

        Parameters
        ----------
        file: str, bytes, int, PathLike
            File with white-space separated tokens.
        cutoff : Cutoff, optional
            Frequency cutoff or target size to restrict vocabulary size. Defaults to
            minimum frequency cutoff of 30.
        indexer : FastTextIndexer, optional
            Subword indexer to use for the vocabulary. Defaults to an indexer
            with 2,000,000 buckets with range 3-6.

        Returns
        -------
        (vocab, counts) : Tuple[FastTextVocab, List[int]]
            Tuple containing the Vocabulary as first item and counts of in-vocabulary items
            as the second item.

        Raises
        ------
        AssertionError
            If the indexer is not a FastTextIndexer.
        """
        assert indexer is None or isinstance(indexer, FastTextIndexer)
        cnt = _count_words(file)
        if cutoff is None:
            cutoff = Cutoff(30, mode='min_freq')
        words, counts = _filter_and_sort(cnt, cutoff)
        return FastTextVocab(words, indexer), counts

    def to_explicit(self) -> 'ExplicitVocab':
        """
        Returns a Vocabulary with explicit storage built from this vocab.

        Returns
        -------
        explicit_vocab : ExplicitVocab
            The converted vocabulary.
        """
        return _to_explicit(self)

    @property
    def subword_indexer(self) -> FastTextIndexer:
        return self._indexer

    @property
    def words(self) -> list:
        return self._words

    @property
    def word_index(self) -> dict:
        return self._index

    @staticmethod
    def read_chunk(file: BinaryIO) -> 'FastTextVocab':
        length, min_n, max_n, buckets = _read_binary(file, "<QIII")
        words, index = _read_items(file, length)
        indexer = FastTextIndexer(buckets, min_n, max_n)
        return FastTextVocab(words, indexer, index)

    def write_chunk(self, file: BinaryIO):
        _write_bucket_vocab(self, file)

    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.FastTextSubwordVocab

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not isinstance(other.subword_indexer, type(self.subword_indexer)):
            return False
        if self.subword_indexer.idx_bound != other.subword_indexer.idx_bound:
            return False
        return super(FastTextVocab, self).__eq__(other)


class ExplicitVocab(Chunk, SubwordVocab):
    """
    A vocabulary with explicitly stored n-grams.
    """
    def __init__(self,
                 words: List[str],
                 indexer: ExplicitIndexer,
                 index: Dict[str, int] = None):
        """
        Initialize an ExplicitVocab.

        Initializes the vocabulary with the given words, subword indexer and an
        optional word index.

        If no index is given, the nth word in the `words` list is assigned
        index `n`. The word list cannot contain duplicate entries and it needs
        to be of same length as the index.

        Parameters
        ----------
        words : List[str]
            List of unique words
        indexer : ExplicitIndexer
            Subword indexer to use for the vocabulary.
        index : Dict[str, int], optional
            Dictionary providing a word -> index mapping.

        Raises
        ------
        ValueError
            if the length of ``index`` and ``word`` doesn't match.
        AssertionError
            If the indexer is not an ExplicitIndexer.

        See Also
        --------
        :class:`.ExplicitIndexer`
        """
        assert isinstance(indexer, ExplicitIndexer)
        super().__init__()
        self._index = _validate_words_and_create_index(words, index)
        self._words = words
        self._indexer = indexer

    @staticmethod
    def from_corpus(file: Union[str, bytes, int, PathLike],
                    ngram_range=(3, 6),
                    token_cutoff: Optional[Cutoff] = None,
                    ngram_cutoff: Optional[Cutoff] = None):
        """
        Build an ExplicitVocab from a corpus.

        Parameters
        ----------
        file: str, bytes, int, PathLike
            File with white-space separated tokens.
        ngram_range : Tuple[int, int]
            Specifies the n-gram range for the indexer.
        token_cutoff : Cutoff, optional
            Frequency cutoff or target size to restrict token vocabulary size. Defaults to
            minimum frequency cutoff of 30.
        ngram_cutoff : Cutoff, optional
            Frequency cutoff or target size to restrict ngram vocabulary size. Defaults to
            minimum frequency cutoff of 30.

        Returns
        -------
        (vocab, counts) : Tuple[FastTextVocab, List[int], List[int]]
            Tuple containing the Vocabulary as first item, counts of in-vocabulary tokens
            as the second item and in-vocabulary ngram counts as the last item.
        """
        min_n, max_n = ngram_range
        cnt = _count_words(file)
        ngram_cnt = collections.Counter()
        for word, count in cnt.items():
            for ngram in word_ngrams(word, min_n, max_n):
                ngram_cnt[ngram] += count
        words, tok_cnt = _filter_and_sort(cnt, token_cutoff)
        ngrams, ngram_cnt = _filter_and_sort(ngram_cnt, ngram_cutoff)
        indexer = ExplicitIndexer(ngrams, ngram_range=ngram_range)
        return ExplicitVocab(words, indexer), tok_cnt, ngram_cnt

    @property
    def words(self) -> list:
        return self._words

    @property
    def word_index(self) -> dict:
        return self._index

    @property
    def subword_indexer(self) -> ExplicitIndexer:
        return self._indexer

    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.ExplicitSubwordVocab

    @staticmethod
    def read_chunk(file: BinaryIO) -> 'ExplicitVocab':
        length, ngram_length, min_n, max_n = _read_binary(file, "<QQII")
        words, word_index = _read_items(file, length)
        ngrams, ngram_index = _read_items(file, ngram_length, indices=True)
        indexer = ExplicitIndexer(ngrams, (min_n, max_n), ngram_index)
        return ExplicitVocab(words, indexer, word_index)

    def write_chunk(self, file) -> None:
        chunk_length = _calculate_serialized_size(self.words)
        chunk_length += _calculate_serialized_size(self.subword_indexer.ngrams)
        min_n_max_n_size = struct.calcsize("<II")
        chunk_length += min_n_max_n_size
        chunk_header = (int(self.chunk_identifier()), chunk_length,
                        len(self.words), len(self.subword_indexer.ngrams),
                        self.min_n, self.max_n)
        _write_binary(file, "<IQQQII", *chunk_header)
        _write_words_binary((bytes(word, "utf-8") for word in self.words),
                            file)
        for ngram in self.subword_indexer.ngrams:
            b_ngram = ngram.encode("utf-8")
            _write_binary(file, "<I", len(b_ngram))
            file.write(b_ngram)
            _write_binary(file, "<Q", self.subword_indexer.ngram_index[ngram])

    def __eq__(self, other):
        if not isinstance(other, ExplicitVocab):
            return False
        if not isinstance(other.subword_indexer, ExplicitIndexer):
            return False
        if self.subword_indexer.idx_bound != other.subword_indexer.idx_bound:
            return False
        if self.subword_indexer.ngrams != other.subword_indexer.ngrams:
            return False
        if self.subword_indexer.ngram_index != other.subword_indexer.ngram_index:
            return False
        return super(ExplicitVocab, self).__eq__(other)


def load_finalfusion_bucket_vocab(file: Union[str, bytes, int, PathLike]
                                  ) -> FinalfusionBucketVocab:
    """
    Load a FinalfusionBucketVocab from the given finalfusion file.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to file containing a FinalfusionBucketVocab chunk.

    Returns
    -------
    vocab : FinalfusionBucketVocab
        Returns the first FinalfusionBucketVocab in the file.
    """
    with open(file, "rb") as inf:
        chunk = find_chunk(inf, [ChunkIdentifier.BucketSubwordVocab])
        if chunk is None:
            raise ValueError('File did not contain a FinalfusionBucketVocab}')
        return FinalfusionBucketVocab.read_chunk(inf)


def load_fasttext_vocab(file: Union[str, bytes, int, PathLike]
                        ) -> FastTextVocab:
    """
    Load a FastTextVocab from the given finalfusion file.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to file containing a FastTextVocab chunk.

    Returns
    -------
    vocab : FastTextVocab
        Returns the first FastTextVocab in the file.
    """
    with open(file, "rb") as inf:
        chunk = find_chunk(inf, [ChunkIdentifier.FastTextSubwordVocab])
        if chunk is None:
            raise ValueError('File did not contain a FastTextVocab}')
        return FastTextVocab.read_chunk(inf)


def load_explicit_vocab(file: Union[str, bytes, int, PathLike]
                        ) -> ExplicitVocab:
    """
    Load a ExplicitVocab from the given finalfusion file.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to file containing a ExplicitVocab chunk.

    Returns
    -------
    vocab : ExplicitVocab
        Returns the first ExplicitVocab in the file.
    """
    with open(file, "rb") as inf:
        chunk = find_chunk(inf, [ChunkIdentifier.ExplicitSubwordVocab])
        if chunk is None:
            raise ValueError('File did not contain a ExplicitVocab}')
        return ExplicitVocab.read_chunk(inf)


def _to_explicit(vocab: Union[FinalfusionBucketVocab, FastTextVocab]
                 ) -> 'ExplicitVocab':
    """
    Convert a bucket vocabulary to an explicit vocab.
    :return: ExplicitVocab
    """
    ngram_index = dict()
    idx_index = dict()
    ngrams = []
    for word in vocab.words:
        token_ngrams = vocab.subwords(word)
        for ngram in token_ngrams:
            if ngram not in ngram_index:
                ngrams.append(ngram)
                idx = vocab.subword_indexer(ngram)
                if idx not in idx_index:
                    idx_index[idx] = len(idx_index)
                ngram_index[ngram] = idx_index[idx]
    indexer = ExplicitIndexer(ngrams, (vocab.min_n, vocab.max_n), ngram_index)
    return ExplicitVocab(vocab.words, indexer, vocab.word_index)


def _write_bucket_vocab(vocab: Union[FinalfusionBucketVocab, FastTextVocab],
                        file: BinaryIO):
    min_n_max_n_size = struct.calcsize("<II")
    buckets_size = struct.calcsize("<I")
    chunk_length = _calculate_serialized_size(vocab.words)
    chunk_length += min_n_max_n_size
    chunk_length += buckets_size

    chunk_id = vocab.chunk_identifier()
    if chunk_id == ChunkIdentifier.FastTextSubwordVocab:
        buckets = vocab.subword_indexer.idx_bound
    else:
        buckets = vocab.subword_indexer.buckets_exp

    chunk_header = (int(chunk_id), chunk_length, len(vocab.words), vocab.min_n,
                    vocab.max_n, buckets)
    _write_binary(file, "<IQQIII", *chunk_header)
    _write_words_binary((bytes(word, "utf-8") for word in vocab.words), file)


__all__ = [
    'SubwordVocab', 'FastTextVocab', 'FinalfusionBucketVocab', 'ExplicitVocab',
    'load_explicit_vocab', 'load_fasttext_vocab',
    'load_finalfusion_bucket_vocab'
]
