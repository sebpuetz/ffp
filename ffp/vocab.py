"""
Finalfusion vocabularies.
"""
import abc
import collections
import operator
import struct
from typing import List, Optional, Dict, Tuple, IO, Iterable, Any, Union, Counter

import ffp.io
import ffp.subwords


class Cutoff:  # pylint: disable=too-few-public-methods
    """
    Frequency Cutoff

    Defines how a vocabulary is sized, if mode is 'min_freq', items with frequency lower than
    `cutoff` are discarded. If mode is 'target_size', the number of items will be smaller than or
    equal to `cutoff`, discarding items at the next frequency-boundary.
    """
    def __init__(self, cutoff: int, mode: str = "min_freq"):
        self.cutoff = cutoff
        self.mode = mode

    @property
    def mode(self) -> str:
        """
        Return the cutoff mode, one of "min_freq" or "target_size".
        :return: The cutoff mode
        """
        return "min_freq" if self._min_freq else "target_size"

    @mode.setter
    def mode(self, mode: str):
        if mode.lower() == "min_freq":
            self._min_freq = True
        elif mode.lower() == "target_size":
            self._min_freq = False
        else:
            raise ValueError(
                "Unknown cutoff mode, expected 'min_freq' or 'target_size' but got: "
                + mode)


class Vocab(ffp.io.Chunk):
    """
    Common interface to finalfusion vocabularies.
    """
    def __init__(self,
                 words: List[str],
                 index: Optional[Dict[str, int]] = None):
        if index is None:
            index = dict((word, idx) for idx, word in enumerate(words))
        if len(index) != len(words):
            raise ValueError("Words and index need to have same length")
        self._index = index
        self._words = words

    @property
    def words(self) -> list:
        """
        Get the list of in-vocabulary words
        :return: list of in-vocabulary words
        """
        return self._words

    @property
    def word_index(self) -> dict:
        """
        Get the dict holding the word-index mapping.
        :return: dict holding the word-index mapping
        """
        return self._index

    @property
    def idx_bound(self) -> int:
        """
        Get the exclusive upper bound of indices this vocabulary covers.
        :return: exclusive upper bound of indices
        """
        return len(self)

    @abc.abstractmethod
    def idx(self, item: str, default: Union[list, int, None] = None
            ) -> Optional[Union[list, int]]:
        """
        Lookup the indices for the given query item.

        This lookup does not raise an exception if the vocab can't produce indices.
        :param item: the query item
        :param default: fall-back return value if no indices can be produced.
        :return: An integer if there is a single index for an in-vocab item,
        a list if the vocab is able to produce subword indices, `default` if
        the vocab is unable to provide any indices, None if not specified.
        """
    def __getitem__(self, item: str) -> Union[list, int]:
        return self.word_index[item]

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return self.word_index.get(item) is not None
        if hasattr(item, "__iter__"):
            return all(w in self for w in item)
        return False

    def __iter__(self) -> Iterable[str]:
        return iter(self.words)

    def __len__(self) -> int:
        return len(self.word_index)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.words != other.words:
            return False
        if self.word_index != other.word_index:
            return False
        return True

    @staticmethod
    def _write_words_binary(b_words: Iterable[bytes], file: IO[bytes]):
        """
        Write the words as bytes to the given file. Each word's bytes are preceded by the number
        of bytes.
        :param b_words:
        :param file:
        """
        for word in b_words:
            file.write(struct.pack("<I", len(word)))
            file.write(word)

    @staticmethod
    def _read_items(file: IO[bytes], length: int,
                    indices=False) -> Tuple[List[str], Dict[str, int]]:
        """
        Read items from a vocabulary chunk.
        :param file: the file containing the chunk
        :param length: the number of items to read
        :param indices: whether each item is followed by an int specifying index.
        :return: A tuple containing the list of items and a dict, mapping each item to an index.
        """
        items = []
        item_index = {}
        for _ in range(length):
            item_length = struct.unpack("<I", file.read(4))[0]
            word = file.read(item_length).decode("utf-8")
            items.append(word)
            if indices:
                item_index[word] = struct.unpack(
                    "<Q", file.read(struct.calcsize("<Q")))[0]
            else:
                item_index[word] = len(item_index)
        return items, item_index

    @staticmethod
    def _count_words(filename) -> Counter:
        cnt = collections.Counter()
        with open(filename) as inf:
            for line in inf:
                for word in line.strip().split():
                    cnt[word] += 1
        return cnt

    @staticmethod
    def _filter_and_sort(cnt: Counter, cutoff: Cutoff):
        cutoff_v = cutoff.cutoff

        def cmp(tup):
            return tup[1] >= cutoff_v

        if cutoff.mode == "min_freq":
            items = sorted(filter(cmp, cnt.items()),
                           key=operator.itemgetter(1, 0),
                           reverse=True)
            if not items:
                return [], []
            keys, cnt = zip(*items)
        else:
            keys, cnt = zip(*sorted(
                cnt.items(), key=operator.itemgetter(1, 0), reverse=True))
            if cutoff_v == 0:
                return [], []
            # cutoff is size, but used as idx
            cutoff_v -= 1
            if cutoff_v <= len(cnt) - 2:
                cnt_at_target = cnt[cutoff_v]
                cnt_after_target = cnt[cutoff_v + 1]
                if cnt_at_target == cnt_after_target:
                    while cutoff_v > 0 and cnt[cutoff_v] == cnt_after_target:
                        cutoff_v -= 1
            keys = keys[:cutoff_v + 1]
            cnt = cnt[:cutoff_v + 1]
        return list(keys), list(cnt)


class SubwordVocab(Vocab):
    """
    Vocabulary type that offers subword lookups.
    """
    def __init__(self,
                 words,
                 indexer: Union[ffp.subwords.FinalfusionHashIndexer, ffp.
                                subwords.FastTextIndexer],
                 index: Optional[Dict[str, int]] = None):
        super().__init__(words, index)
        self._indexer = indexer

    def idx(self, item, default=None):
        idx = self.word_index.get(item)
        if idx is not None:
            return idx
        subwords = self.subword_indices(item)
        if subwords:
            return subwords
        return default

    @property
    def idx_bound(self) -> int:
        return len(self) + self.indexer.idx_bound

    @property
    def min_n(self) -> int:
        """
        Get the lower bound of the range of extracted n-grams.
        :return: lower bound of n-gram range.
        """
        return self.indexer.min_n

    @property
    def max_n(self) -> int:
        """
        Get the upper bound of the range of extracted n-grams.
        :return: upper bound of n-gram range.
        """
        return self.indexer.max_n

    @property
    def indexer(
            self
    ) -> Union[ffp.subwords.ExplicitIndexer, ffp.subwords.
               FinalfusionHashIndexer, ffp.subwords.FastTextIndexer]:
        """
        Get this vocab's indexer. The indexer produces indices for given n-grams.

        In case of bucket vocabularies, this is a hash-based indexer. For explicit subword
        vocabularies, this is the vocabulary itself since it holds a dict mapping ngrams to
        indices.
        """
        return self._indexer

    def subwords(self, item: str, bracket=True) -> List[str]:
        """
        Get the n-grams of the given item as a list.
        :param item: the query item
        :param bracket: whether to bracket the item with '<' and '>'
        :return: The list of n-grams.
        """
        return ffp.subwords.word_ngrams(item, self.min_n, self.max_n, bracket)

    def subword_indices(self, item: str, bracket=True) -> List[int]:
        """
        Get the subword indices for the given item.
        :param item: the query item
        :param bracket: whether to bracket the item with '<' and '>'
        :return: A list of indices
        """
        return self.indexer.subword_indices(item,
                                            offset=len(self.words),
                                            bracket=bracket)

    def __getitem__(self, item: str) -> Union[int, List[int]]:
        idx = self.word_index.get(item)
        if idx is not None:
            return idx
        subwords = self.subword_indices(item)
        if subwords:
            return subwords
        raise KeyError("No indices foud for " + item)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.min_n != other.min_n:
            return False
        if self.max_n != other.max_n:
            return False
        return super(SubwordVocab, self).__eq__(other)

    @staticmethod
    def chunk_identifier() -> ffp.io.ChunkIdentifier:
        pass

    @staticmethod
    def read_chunk(file: IO[bytes]) -> 'ffp.io.Chunk':
        pass

    def write_chunk(self, file: IO[bytes]):
        pass


class BucketVocab(SubwordVocab):
    """
    Bucket vocabulary, hashes ngrams with an imperfect hashing function.
    """
    def to_explicit(self) -> 'ExplicitVocab':
        """
        Convert a bucket vocabulary to an explicit vocab.
        :return: ExplicitVocab
        """
        ngram_index = dict()
        idx_index = dict()
        ngrams = []
        for word in self.words:
            token_ngrams = self.subwords(word)
            for ngram in token_ngrams:
                if ngram not in ngram_index:
                    ngrams.append(ngram)
                    idx = self.indexer(ngram)
                    if idx not in idx_index:
                        idx_index[idx] = len(idx_index)
                    ngram_index[ngram] = idx_index[idx]
        indexer = ffp.subwords.ExplicitIndexer(ngrams,
                                               (self.min_n, self.max_n),
                                               ngram_index)
        return ExplicitVocab(self.words, indexer, self.word_index)

    @staticmethod
    def chunk_identifier() -> ffp.io.ChunkIdentifier:
        pass

    @staticmethod
    def read_chunk(file: IO[bytes]) -> 'ffp.io.Chunk':
        pass

    def write_chunk(self, file):
        chunk_id = self.chunk_identifier()
        file.write(struct.pack("<I", int(chunk_id)))
        b_words = [bytes(word, "utf-8") for word in self.words]
        chunk_length = struct.calcsize(
            "<QIII") + len(b_words) * struct.calcsize("<I") + sum(
                [len(word) for word in b_words])
        if chunk_id == ffp.io.ChunkIdentifier.FastTextSubwordVocab:
            buckets = self.indexer.idx_bound
        else:
            buckets = self.indexer.buckets_exp
        file.write(
            struct.pack("<QQIII", chunk_length, len(b_words), self.min_n,
                        self.max_n, buckets))
        self._write_words_binary(b_words, file)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not isinstance(other.indexer, type(self.indexer)):
            return False
        if self.indexer.idx_bound != other.indexer.idx_bound:
            return False
        return super(BucketVocab, self).__eq__(other)


class FinalfusionBucketVocab(BucketVocab):
    """
    Finalfusion Bucket Vocabulary.
    """
    def __init__(self, words, indexer=None, index=None):
        if indexer is None:
            indexer = ffp.subwords.FinalfusionHashIndexer(21)
        super().__init__(words, indexer, index)

    @staticmethod
    def from_corpus(
            filename,
            cutoff=Cutoff(30, mode='min_freq'),
            indexer: Optional[ffp.subwords.FinalfusionHashIndexer] = None,
    ) -> Tuple['ffp.subwords.FinalfusionBucketVocab', List[int]]:
        """
        Construct a Finalfusion Bucket Vocabulary from the given corpus.
        :param filename: file containing white-space seperated tokens.
        :param cutoff: set the number of tokens in the vocabulary, either as a frequency
        cutoff or as a target size
        :param indexer: Indexer to be used
        :param ngram_range: bounds of extracted n-gram range
        :return: Tuple containing FinalfusionBucketVocab and in-vocab token counts
        """
        assert isinstance(
            indexer, ffp.subwords.FinalfusionHashIndexer) or indexer is None
        cnt = FinalfusionBucketVocab._count_words(filename)
        words, counts = FinalfusionBucketVocab._filter_and_sort(cnt, cutoff)
        return FinalfusionBucketVocab(words, indexer), counts

    @staticmethod
    def read_chunk(file) -> 'FinalfusionBucketVocab':
        length, min_n, max_n, buckets = struct.unpack(
            "<QIII", file.read(struct.calcsize("<QIII")))
        words, index = FinalfusionBucketVocab._read_items(file, length)
        indexer = ffp.subwords.FinalfusionHashIndexer(buckets, min_n, max_n)
        return FinalfusionBucketVocab(words, indexer, index)

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.BucketSubwordVocab


class FastTextVocab(BucketVocab):
    """
    FastText vocabulary
    """
    def __init__(self, words, indexer=None, index=None):
        if indexer is None:
            indexer = ffp.subwords.FastTextIndexer(2000000)
        super().__init__(words, indexer, index)

    @staticmethod
    def from_corpus(
            filename,
            cutoff: Cutoff = Cutoff(30, mode='min_freq'),
            indexer: Optional[ffp.subwords.FastTextIndexer] = None,
    ) -> Tuple['FastTextVocab', List[int]]:
        """
        Construct a fastText vocabulary from the given corpus.
        :param filename: file containing white-space seperated tokens.
        :param cutoff: set the number of tokens in the vocabulary, either as a frequency
        cutoff or as a target size
        :param indexer: Indexer to be used
        :param ngram_range: bounds of extracted n-gram range
        :return: Tuple containing FastTextVocab and in-vocab token counts
        """
        assert isinstance(indexer,
                          ffp.subwords.FastTextIndexer) or indexer is None
        cnt = FastTextVocab._count_words(filename)
        words, counts = FastTextVocab._filter_and_sort(cnt, cutoff)
        return FastTextVocab(words, indexer), counts

    @staticmethod
    def read_chunk(file) -> 'FastTextVocab':
        length, min_n, max_n, buckets = struct.unpack(
            "<QIII", file.read(struct.calcsize("<QIII")))
        words, index = FastTextVocab._read_items(file, length)
        indexer = ffp.subwords.FastTextIndexer(buckets, min_n, max_n)
        return FastTextVocab(words, indexer, index)

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.FastTextSubwordVocab


class ExplicitVocab(SubwordVocab):
    """
    A vocabulary with explicitly stored n-grams.
    """
    def __init__(self, words, indexer, index=None):
        assert isinstance(indexer, ffp.subwords.ExplicitIndexer)
        super().__init__(words, indexer, index)

    @staticmethod
    def from_corpus(filename,
                    ngram_range=(3, 6),
                    token_cutoff: Cutoff = Cutoff(30, mode='min_freq'),
                    ngram_cutoff: Cutoff = Cutoff(30, mode='min_freq')):
        """
        Construct a explicit vocabulary from the given corpus.
        :param filename: file containing white-space seperated tokens.
        :param ngram_range: bounds of extracted n-gram range
        :param token_cutoff: set the number of tokens in the vocabulary, either as a frequency
        cutoff or as a target size
        :param ngram_cutoff: set the number of ngrams in the vocabulary, either as a frequency
        cutoff or as a target size
        :return: a tuple containing the vocabulary, token counts and n-gram counts
        """
        min_n, max_n = ngram_range
        cnt = ExplicitVocab._count_words(filename)
        ngram_cnt = collections.Counter()
        for word, count in cnt.items():
            for ngram in ffp.subwords.word_ngrams(word, min_n, max_n):
                ngram_cnt[ngram] += count
        words, tok_cnt = ExplicitVocab._filter_and_sort(cnt, token_cutoff)
        ngrams, ngram_cnt = ExplicitVocab._filter_and_sort(
            ngram_cnt, ngram_cutoff)
        indexer = ffp.subwords.ExplicitIndexer(ngrams, ngram_range=ngram_range)
        return ExplicitVocab(words, indexer), tok_cnt, ngram_cnt

    @property
    def indexer(self):
        return self._indexer

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.ExplicitSubwordVocab

    @staticmethod
    def read_chunk(file) -> 'ExplicitVocab':
        length, ngram_length, min_n, max_n = struct.unpack(
            "<QQII", file.read(struct.calcsize("<QQII")))
        words, word_index = ExplicitVocab._read_items(file, length)
        ngrams, ngram_index = ExplicitVocab._read_items(file,
                                                        ngram_length,
                                                        indices=True)
        indexer = ffp.subwords.ExplicitIndexer(ngrams, (min_n, max_n),
                                               ngram_index)
        return ExplicitVocab(words, indexer, word_index)

    def write_chunk(self, file) -> None:
        chunk_id = self.chunk_identifier()
        file.write(struct.pack("<I", int(chunk_id)))

        b_words = [bytes(word, "utf-8") for word in self.words]
        b_len_words = len(b_words) * struct.calcsize("<I") + sum(
            [len(word) for word in b_words])

        b_ngrams = [bytes(ngram, "utf-8") for ngram in self.indexer]
        b_len_ngrams = len(b_ngrams) * struct.calcsize("<I") + sum(
            [len(ngram) for ngram in b_ngrams])

        chunk_length = struct.calcsize("<QQII") + b_len_words + b_len_ngrams

        file.write(
            struct.pack("<QQQII", chunk_length, len(b_words), len(b_ngrams),
                        self.min_n, self.max_n))

        self._write_words_binary(b_words, file)
        for i, ngram in enumerate(b_ngrams):
            file.write(struct.pack("<I", len(ngram)))
            file.write(ngram)
            file.write(
                struct.pack("<Q",
                            self.indexer.ngram_index[self.indexer.ngrams[i]]))


class SimpleVocab(Vocab):
    """
    Simple vocabulary without subword indices.
    """
    @staticmethod
    def from_corpus(filename, cutoff: Cutoff = Cutoff(30, mode="min_freq")):
        """
        Construct a simple vocabulary from the given corpus.
        :param filename: file
        :param cutoff: set the number of tokens in the vocabulary, either as a frequency
        cutoff or as a target size
        :return: Tuple containing SimpleVocab and in-vocab token counts
        """
        cnt = SimpleVocab._count_words(filename)
        words, cnt = SimpleVocab._filter_and_sort(cnt, cutoff)
        return SimpleVocab(words), cnt

    @staticmethod
    def read_chunk(file) -> 'SimpleVocab':
        length = struct.unpack("<Q", file.read(8))[0]
        words, index = SimpleVocab._read_items(file, length)
        return SimpleVocab(words, index)

    def write_chunk(self, file):
        file.write(struct.pack("<I", int(self.chunk_identifier())))
        b_words = [bytes(word, "utf-8") for word in self.words]
        chunk_length = struct.calcsize(
            "<Q") + len(b_words) * struct.calcsize("<I") + sum(
                [len(word) for word in b_words])
        file.write(struct.pack("<QQ", chunk_length, len(b_words)))
        self._write_words_binary(b_words, file)

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.SimpleVocab

    def __getitem__(self, item):
        return self.word_index[item]

    def idx(self, item, default=None):
        return self.word_index.get(item, default)


def load_vocab(path: str) -> Vocab:
    """
    Read a vocabulary from the given finalfusion file.
    :param path: Path to a file containing a finalfusion vocabulary.
    :return: The first vocabulary found in the file.
    """
    vocab_chunks = [
        ffp.io.ChunkIdentifier.SimpleVocab,
        ffp.io.ChunkIdentifier.BucketSubwordVocab,
        ffp.io.ChunkIdentifier.ExplicitSubwordVocab,
        ffp.io.ChunkIdentifier.FastTextSubwordVocab,
    ]
    vocab = _read(path, vocab_chunks)
    if vocab is None:
        raise IOError("File did not contain a supported vocabulary")
    return vocab


def load_simple_vocab(path: str) -> SimpleVocab:
    """
    Load a SimpleVocab from the given finalfusion file.
    :param path: Path to a file containing a finalfusion SimpleVocab.
    :return: The first SimpleVocab found in the file.
    """
    vocab = _read(path, [ffp.io.ChunkIdentifier.SimpleVocab])
    if vocab is None:
        raise IOError("File did not contain a vocabulary")
    return vocab


def load_subword_vocab(path: str) -> SubwordVocab:
    """
    Load a SubwordVocab from the given finalfusion file.
    :param path: Path to a file containing a finalfusion SimpleVocab.
    :return: The first SimpleVocab found in the file.
    """
    vocab_chunks = [
        ffp.io.ChunkIdentifier.BucketSubwordVocab,
        ffp.io.ChunkIdentifier.ExplicitSubwordVocab,
        ffp.io.ChunkIdentifier.FastTextSubwordVocab
    ]
    vocab = _read(path, vocab_chunks)
    if vocab is None:
        raise IOError("File did not contain a SubwordVocab")
    return vocab


def load_bucket_vocab(path: str) -> BucketVocab:
    """
    Load a BucketVocab from the given finalfusion file.
    :param path: Path to a file containing a finalfusion SimpleVocab.
    :return: The first SimpleVocab found in the file.
    """
    vocab_chunks = [
        ffp.io.ChunkIdentifier.BucketSubwordVocab,
        ffp.io.ChunkIdentifier.FastTextSubwordVocab
    ]
    vocab = _read(path, vocab_chunks)
    if vocab is None:
        raise IOError("File did not contain a BucketVocab")
    return vocab


def load_finalfusion_bucket_vocab(path: str) -> FinalfusionBucketVocab:
    """
    Load a FinalfusionBucketVocab from the given finalfusion file.
    :param path: Path to a file containing a finalfusion SimpleVocab.
    :return: The first SimpleVocab found in the file.
    """
    vocab = _read(path, [ffp.io.ChunkIdentifier.BucketSubwordVocab])
    if vocab is None:
        raise IOError("File did not contain a SubwordVocab")
    return vocab


def load_fasttext_vocab(path) -> FastTextVocab:
    """
    Load a FastTextVocab from the given finalfusion file.
    :param path: Path to a file containing a finalfusion SimpleVocab.
    :return: The first SimpleVocab found in the file.
    """
    vocab = _read(path, [ffp.io.ChunkIdentifier.FastTextSubwordVocab])
    if vocab is None:
        raise IOError("File did not contain a vocabulary")
    return vocab


def load_explicit_vocab(path: str) -> ExplicitVocab:
    """
    Load an ExplicitVocab from the given finalfusion file.
    :param path: Path to a file containing a finalfusion SimpleVocab.
    :return: The first SimpleVocab found in the file.
    """
    vocab = _read(path, [ffp.io.ChunkIdentifier.ExplicitSubwordVocab])
    if vocab is None:
        raise IOError("File did not contain an ExplicitVocab")
    return vocab


def _read(path: str, target: List[ffp.io.ChunkIdentifier]):
    """
    Read the first chunk specified in `target` from `filename`.
    :param path: filename
    :param target: List of target chunks
    :return: Vocab
    """
    with open(path, "rb") as file:
        chunk = ffp.io.find_chunk(file, target)
        if chunk is None:
            return None
        if chunk == ffp.io.ChunkIdentifier.SimpleVocab:
            return SimpleVocab.read_chunk(file)
        if chunk == ffp.io.ChunkIdentifier.BucketSubwordVocab:
            return FinalfusionBucketVocab.read_chunk(file)
        if chunk == ffp.io.ChunkIdentifier.ExplicitSubwordVocab:
            return ExplicitVocab.read_chunk(file)
        if chunk == ffp.io.ChunkIdentifier.FastTextSubwordVocab:
            return FastTextVocab.read_chunk(file)
        raise IOError("unknown vocab type: " + str(chunk))
