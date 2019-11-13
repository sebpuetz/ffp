"""
Finalfusion vocabularies.
"""
import struct
from abc import abstractmethod
from typing import List, Optional, Dict, Tuple, IO, Iterable, Any, Union

from ffp import io
from ffp.vocab_rs import FastTextIndexer, FinalfusionHashIndexer, \
    word_ngrams as _word_ngrams


def word_ngrams(word, min_n=3, max_n=6, bracket=True) -> List[str]:
    """
    Get the ngrams for the given word.
    :param word: word
    :param min_n: lower bound of n-gram range
    :param max_n: upper bound of n-gram range
    :param bracket: whether to bracket the word with '<' and '>'
    :return: list of n-grams
    """
    return _word_ngrams(word, min_n, max_n, bracket)


class Vocab(io.Chunk):
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
            io.ChunkIdentifier.SimpleVocab,
            io.ChunkIdentifier.BucketSubwordVocab,
            io.ChunkIdentifier.FastTextSubwordVocab,
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
            header = io.Header([self.chunk_identifier()])
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

    @abstractmethod
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
    def _read(filename: str, target: List[io.ChunkIdentifier]):
        """
        Read the first chunk specified in `target` from `filename`.
        :param filename: filename
        :param target: List of target chunks
        :return: Vocab
        """
        with open(filename, "rb") as file:
            chunk = io.find_chunk(file, target)
            if chunk is None:
                return None
            if chunk == io.ChunkIdentifier.SimpleVocab:
                return SimpleVocab.read_chunk(file)
            if chunk == io.ChunkIdentifier.BucketSubwordVocab:
                return FinalfusionBucketVocab.read_chunk(file)
            if chunk == io.ChunkIdentifier.FastTextSubwordVocab:
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
                 indexer: Union[FinalfusionHashIndexer, FastTextIndexer],
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
            io.ChunkIdentifier.BucketSubwordVocab,
            io.ChunkIdentifier.ExplicitSubwordVocab,
            io.ChunkIdentifier.FastTextSubwordVocab
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
    def indexer(self) -> Union[FinalfusionHashIndexer, FastTextIndexer]:
        """
        Get this vocab's indexer. The indexer produces indices for given n-grams.

        In case of bucket vocabularies, this is a hash-based indexer. For explicit subword
        vocabularies, this is the vocabulary itself since it holds a dict mapping ngrams to
        indices.
        """
        return self._indexer

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
    def chunk_identifier() -> io.ChunkIdentifier:
        pass

    @staticmethod
    def read_chunk(file: IO[bytes]) -> 'io.Chunk':
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
            io.ChunkIdentifier.BucketSubwordVocab,
            io.ChunkIdentifier.FastTextSubwordVocab
        ]
        vocab = BucketVocab._read(filename, vocab_chunks)
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

    @staticmethod
    def chunk_identifier() -> io.ChunkIdentifier:
        pass

    @staticmethod
    def read_chunk(file: IO[bytes]) -> 'io.Chunk':
        pass

    def write_chunk(self, file):
        chunk_id = self.chunk_identifier()
        file.write(struct.pack("<I", int(chunk_id)))
        b_words = [bytes(word, "utf-8") for word in self.words]
        chunk_length = struct.calcsize(
            "<QIII") + len(b_words) * struct.calcsize("<I") + sum(
                [len(word) for word in b_words])
        if chunk_id == io.ChunkIdentifier.FastTextSubwordVocab:
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
            indexer = FinalfusionHashIndexer(21)
        super().__init__(words, indexer, ngram_range, index)

    @staticmethod
    def read(filename):
        vocab = FinalfusionBucketVocab._read(
            filename, [io.ChunkIdentifier.BucketSubwordVocab])
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

    @staticmethod
    def read_chunk(file) -> 'FinalfusionBucketVocab':
        length, min_n, max_n, buckets = struct.unpack(
            "<QIII", file.read(struct.calcsize("<QIII")))
        words, index = FinalfusionBucketVocab._read_items(file, length)
        ngram_range = (min_n, max_n)
        indexer = FinalfusionHashIndexer(buckets)
        return FinalfusionBucketVocab(words, indexer, ngram_range, index)

    @staticmethod
    def chunk_identifier():
        return io.ChunkIdentifier.BucketSubwordVocab


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
            indexer = FastTextIndexer(2000000)
        super().__init__(words, indexer, ngram_range, index)

    @staticmethod
    def read(filename):
        vocab = FastTextVocab._read(filename,
                                    [io.ChunkIdentifier.FastTextSubwordVocab])
        if vocab is None:
            raise IOError("File did not contain a vocabulary")
        return vocab

    @staticmethod
    def read_chunk(file) -> 'FastTextVocab':
        length, min_n, max_n, buckets = struct.unpack(
            "<QIII", file.read(struct.calcsize("<QIII")))
        words, index = FastTextVocab._read_items(file, length)
        ngram_range = (min_n, max_n)
        indexer = FastTextIndexer(buckets)
        return FastTextVocab(words, indexer, ngram_range, index)

    @staticmethod
    def chunk_identifier():
        return io.ChunkIdentifier.FastTextSubwordVocab


class SimpleVocab(Vocab):
    """
    Simple vocabulary without subword indices.
    """
    @staticmethod
    def read(filename) -> 'SimpleVocab':
        vocab = SimpleVocab._read(filename, [io.ChunkIdentifier.SimpleVocab])
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
        return io.ChunkIdentifier.SimpleVocab

    def __getitem__(self, item):
        return self.word_index[item]

    def idx(self, item, default=None):
        return self.word_index.get(item, default)
