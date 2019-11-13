"""
Finalfusion vocabularies.
"""
import abc
import struct
from typing import List, Optional, Dict, Tuple, IO, Iterable, Any, Union

import ffp.io
import ffp.subwords


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

    @staticmethod
    def read(filename: str) -> 'Vocab':
        """
        Read a vocabulary from the given finalfusion file.
        :param filename: Path to a file containing a finalfusion vocabulary.
        :return: The first vocabulary found in the file.
        """
        vocab_chunks = [
            ffp.io.ChunkIdentifier.SimpleVocab,
            ffp.io.ChunkIdentifier.BucketSubwordVocab,
            ffp.io.ChunkIdentifier.FastTextSubwordVocab,
        ]
        vocab = Vocab._read(filename, vocab_chunks)
        if vocab is None:
            raise IOError("File did not contain a supported vocabulary")
        return vocab

    def write(self, filename: str):
        """
        Write the vocabulary to the given file in finalfusion format.
        :param filename:
        """
        with open(filename, "wb") as file:
            header = ffp.io.Header([self.chunk_identifier()])
            header.write_chunk(file)
            self.write_chunk(file)

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
    def _read(filename: str, target: List[ffp.io.ChunkIdentifier]):
        """
        Read the first chunk specified in `target` from `filename`.
        :param filename: filename
        :param target: List of target chunks
        :return: Vocab
        """
        with open(filename, "rb") as file:
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


class SubwordVocab(Vocab):
    """
    Vocabulary type that offers subword lookups.
    """
    def __init__(self,
                 words,
                 indexer: Union[ffp.subwords.FinalfusionHashIndexer, ffp.
                                subwords.FastTextIndexer],
                 ngram_range: Tuple[int, int] = (3, 6),
                 index: Optional[Dict[str, int]] = None):
        super().__init__(words, index)
        self._indexer = indexer
        self._min_n, self._max_n = ngram_range

    def idx(self, item, default=None):
        idx = self.word_index.get(item)
        if idx is not None:
            return idx
        subwords = self.subword_indices(item)
        if subwords:
            return subwords
        return default

    @staticmethod
    def read(filename) -> 'SubwordVocab':
        vocab_chunks = [
            ffp.io.ChunkIdentifier.BucketSubwordVocab,
            ffp.io.ChunkIdentifier.ExplicitSubwordVocab,
            ffp.io.ChunkIdentifier.FastTextSubwordVocab
        ]
        vocab = SubwordVocab._read(filename, vocab_chunks)
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

    @property
    def idx_bound(self) -> int:
        return len(self) + self.indexer.idx_bound

    @property
    def min_n(self) -> int:
        """
        Get the lower bound of the range of extracted n-grams.
        :return: lower bound of n-gram range.
        """
        return self._min_n

    @property
    def max_n(self) -> int:
        """
        Get the upper bound of the range of extracted n-grams.
        :return: upper bound of n-gram range.
        """
        return self._max_n

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
                                            min_n=self.min_n,
                                            max_n=self.max_n,
                                            offset=len(self.words),
                                            bracket=bracket)

    def __getitem__(self, item: str) -> int:
        return self.word_index[item]

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
    @staticmethod
    def read(filename) -> 'BucketVocab':
        vocab_chunks = [
            ffp.io.ChunkIdentifier.BucketSubwordVocab,
            ffp.io.ChunkIdentifier.FastTextSubwordVocab
        ]
        vocab = BucketVocab._read(filename, vocab_chunks)
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

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
        indexer = ffp.subwords.ExplicitIndexer(ngrams, ngram_index)
        return ExplicitVocab(self.words, indexer, (self.min_n, self.max_n),
                             self.word_index)

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
    def __init__(self,
                 words,
                 indexer=None,
                 ngram_range: Tuple[int, int] = (3, 6),
                 index=None):
        if indexer is None:
            indexer = ffp.subwords.FinalfusionHashIndexer(21)
        super().__init__(words, indexer, ngram_range, index)

    @staticmethod
    def read(filename):
        vocab = FinalfusionBucketVocab._read(
            filename, [ffp.io.ChunkIdentifier.BucketSubwordVocab])
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

    @staticmethod
    def read_chunk(file) -> 'FinalfusionBucketVocab':
        length, min_n, max_n, buckets = struct.unpack(
            "<QIII", file.read(struct.calcsize("<QIII")))
        words, index = FinalfusionBucketVocab._read_items(file, length)
        ngram_range = (min_n, max_n)
        indexer = ffp.subwords.FinalfusionHashIndexer(buckets)
        return FinalfusionBucketVocab(words, indexer, ngram_range, index)

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.BucketSubwordVocab


class FastTextVocab(BucketVocab):
    """
    FastText vocabulary
    """
    def __init__(self,
                 words,
                 indexer=None,
                 ngram_range: Tuple[int, int] = (3, 6),
                 index=None):
        if indexer is None:
            indexer = ffp.subwords.FastTextIndexer(2000000)
        super().__init__(words, indexer, ngram_range, index)

    @staticmethod
    def read(filename):
        vocab = FastTextVocab._read(
            filename, [ffp.io.ChunkIdentifier.FastTextSubwordVocab])
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

    @staticmethod
    def read_chunk(file) -> 'FastTextVocab':
        length, min_n, max_n, buckets = struct.unpack(
            "<QIII", file.read(struct.calcsize("<QIII")))
        words, index = FastTextVocab._read_items(file, length)
        ngram_range = (min_n, max_n)
        indexer = ffp.subwords.FastTextIndexer(buckets)
        return FastTextVocab(words, indexer, ngram_range, index)

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.FastTextSubwordVocab


class ExplicitVocab(SubwordVocab):
    """
    A vocabulary with explicitly stored n-grams.
    """
    def __init__(self,
                 words,
                 indexer,
                 ngram_range: Tuple[int, int] = (3, 6),
                 index=None):
        assert isinstance(indexer, ffp.subwords.ExplicitIndexer)
        super().__init__(words, indexer, ngram_range, index)

    @staticmethod
    def read(filename) -> 'ExplicitVocab':
        vocab = ExplicitVocab._read(
            filename, [ffp.io.ChunkIdentifier.ExplicitSubwordVocab])
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

    @property
    def indexer(self):
        return self._indexer

    def subword_indices(self, item, bracket=True):
        ngrams = self.subwords(item, bracket)
        return [
            idx + len(self) for idx in (self.indexer(ngram)
                                        for ngram in ngrams) if idx is not None
        ]

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
        indexer = ffp.subwords.ExplicitIndexer(ngrams, ngram_index)
        return ExplicitVocab(words, indexer, (min_n, max_n), word_index)

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
    def read(filename) -> 'SimpleVocab':
        vocab = SimpleVocab._read(filename,
                                  [ffp.io.ChunkIdentifier.SimpleVocab])
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

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
