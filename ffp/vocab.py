"""
Finalfusion vocabularies.
"""
import struct
from abc import abstractmethod
from typing import List, Optional, Dict, Tuple, IO, Iterable, Any, Union

from ffp import io


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
