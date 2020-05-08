"""
Finalfusion SimpleVocab
"""
from os import PathLike
from typing import List, Optional, Dict, Union, BinaryIO

from ffp.vocab.vocab import Vocab, _write_words_binary, _calculate_serialized_size, _read_items
from ffp.vocab.cutoff import Cutoff, _filter_and_sort, _count_words
from ffp.io import ChunkIdentifier, find_chunk, _read_binary, _write_binary, Chunk


class SimpleVocab(Chunk, Vocab):
    """
    Simple vocabulary.

    SimpleVocabs provide a simple string to index mapping and index to string
    mapping. SimpleVocab is also the base type of other vocabulary types.
    """
    def __init__(self,
                 words: List[str],
                 index: Optional[Dict[str, int]] = None):
        """
        Initialize a SimpleVocab.

        Initializes the vocabulary with the given words and optional index. If
        no index is given, the nth word in the `words` list is assigned index
        `n`. The word list cannot contain duplicate entries and it needs to be
        of same length as the index.

        Parameters
        ----------
        words : List[str]
            List of unique words
        index : Optional[Dict[str, int]]
            Dictionary providing an entry -> index mapping.

        Raises
        ------
        ValueError
            if the length of `index` and `word` doesn't match.
        """
        if index is None:
            index = dict((word, idx) for idx, word in enumerate(words))
        if len(index) != len(words):
            raise ValueError("Words and index need to have same length")
        self._index = index
        self._words = words

    @staticmethod
    def from_corpus(file: Union[str, bytes, int, PathLike],
                    cutoff: Cutoff = Cutoff(30, mode="min_freq")):
        """
        Construct a simple vocabulary from the given corpus.

        Parameters
        ----------
        file: str, bytes, int, PathLike
            Path to corpus file
        cutoff : Cutoff
            Frequency cutoff or target size to restrict vocabulary size.

        Returns
        -------
        (vocab, counts) : Tuple[SimpleVocab, List[int]]
            Tuple containing the Vocabulary as first item and counts of in-vocabulary items
            as the second item.
        """
        cnt = _count_words(file)
        words, cnt = _filter_and_sort(cnt, cutoff)
        return SimpleVocab(words), cnt

    @property
    def word_index(self) -> dict:
        return self._index

    @property
    def words(self) -> list:
        return self._words

    @property
    def idx_bound(self) -> int:
        return len(self._index)

    @staticmethod
    def read_chunk(file: BinaryIO) -> 'SimpleVocab':
        length = _read_binary(file, "<Q")[0]
        words, index = _read_items(file, length)
        return SimpleVocab(words, index)

    def write_chunk(self, file: BinaryIO):
        _write_binary(file, "<I", int(self.chunk_identifier()))
        chunk_length = _calculate_serialized_size(self.words)
        _write_binary(file, "<QQ", chunk_length, len(self.words))
        _write_words_binary((bytes(word, "utf-8") for word in self.words),
                            file)

    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.SimpleVocab

    def __getitem__(self, item):
        return self.word_index[item]

    def idx(self, item, default=None):
        return self.word_index.get(item, default)


def load_simple_vocab(file: Union[str, bytes, int, PathLike]) -> SimpleVocab:
    """
    Load a SimpleVocab from the given finalfusion file.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to file containing a SimpleVocab chunk.

    Returns
    -------
    vocab : SimpleVocab
        Returns the first SimpleVocab in the file.
    """
    with open(file, "rb") as inf:
        chunk = find_chunk(inf, [ChunkIdentifier.SimpleVocab])
        if chunk is None:
            raise ValueError('File did not contain a SimpleVocab}')
        return SimpleVocab.read_chunk(inf)


__all__ = ['SimpleVocab', 'load_simple_vocab']
