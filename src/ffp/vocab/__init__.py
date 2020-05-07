"""
Finalfusion vocabularies.
"""
from os import PathLike
from typing import Union

from ffp.io import ChunkIdentifier, find_chunk

from ffp.vocab.simple_vocab import load_simple_vocab, SimpleVocab
from ffp.vocab.subword import load_explicit_vocab, load_fasttext_vocab, \
    load_finalfusion_bucket_vocab, SubwordVocab, FastTextVocab,\
    FinalfusionBucketVocab, ExplicitVocab
from ffp.vocab.vocab import Vocab
from ffp.vocab.cutoff import Cutoff


def load_vocab(file: Union[str, bytes, int, PathLike]) -> Vocab:
    """
    Load a vocabulary from a finalfusion file.

    Loads the first known vocabulary from a finalfusion file.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to file containing a finalfusion vocab chunk.

    Returns
    -------
    vocab : SimpleVocab, FastTextVocab, FinalfusionBucketVocab, ExplicitVocab
        First Vocab in the file.

    Raises
    ------
    ValueError
         If the file did not contain a vocabulary.
    """
    with open(file, "rb") as inf:
        chunk = find_chunk(inf, [
            ChunkIdentifier.SimpleVocab, ChunkIdentifier.FastTextSubwordVocab,
            ChunkIdentifier.ExplicitSubwordVocab,
            ChunkIdentifier.BucketSubwordVocab
        ])
        if chunk is None:
            raise ValueError('File did not contain a vocabulary')
        if chunk == ChunkIdentifier.SimpleVocab:
            return SimpleVocab.read_chunk(inf)
        if chunk == ChunkIdentifier.BucketSubwordVocab:
            return FinalfusionBucketVocab.read_chunk(inf)
        if chunk == ChunkIdentifier.ExplicitSubwordVocab:
            return ExplicitVocab.read_chunk(inf)
        if chunk == ChunkIdentifier.FastTextSubwordVocab:
            return FastTextVocab.read_chunk(inf)
        raise ValueError('Unexpected vocabulary chunk.')


__all__ = [
    'Vocab', 'ExplicitVocab', 'FinalfusionBucketVocab', 'FastTextVocab',
    'SimpleVocab', 'load_finalfusion_bucket_vocab', 'load_fasttext_vocab',
    'load_explicit_vocab', 'load_simple_vocab', 'load_vocab'
]
