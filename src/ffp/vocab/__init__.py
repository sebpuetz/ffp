"""
Finalfusion vocabularies.
"""

from ffp.io import ChunkIdentifier, find_chunk

from ffp.vocab.simple_vocab import load_simple_vocab, SimpleVocab
from ffp.vocab.subword import load_explicit_vocab, load_fasttext_vocab, \
    load_finalfusion_bucket_vocab, SubwordVocab, FastTextVocab,\
    FinalfusionBucketVocab, ExplicitVocab
from ffp.vocab.vocab import Vocab
from ffp.vocab.cutoff import Cutoff


def load_vocab(path: str) -> Vocab:
    """
    Load any vocabulary from a finalfusion file.

    Loads the first known vocabulary from a finalfusion file.

    Parameters
    ----------
    path : str
        Path to file containing a finalfusion vocab chunk.

    Returns
    -------
    vocab : Union[SimpleVocab]
        First vocabulary in the file.

    Raises
    ------
    ValueError
         If the file did not contain a vocabulary.
    """
    with open(path, "rb") as file:
        chunk = find_chunk(file, [
            ChunkIdentifier.SimpleVocab, ChunkIdentifier.FastTextSubwordVocab,
            ChunkIdentifier.ExplicitSubwordVocab,
            ChunkIdentifier.BucketSubwordVocab
        ])
        if chunk is None:
            raise ValueError('File did not contain a vocabulary')
        if chunk == ChunkIdentifier.SimpleVocab:
            return SimpleVocab.read_chunk(file)
        if chunk == ChunkIdentifier.BucketSubwordVocab:
            return FinalfusionBucketVocab.read_chunk(file)
        if chunk == ChunkIdentifier.ExplicitSubwordVocab:
            return ExplicitVocab.read_chunk(file)
        if chunk == ChunkIdentifier.FastTextSubwordVocab:
            return FastTextVocab.read_chunk(file)
        raise ValueError('Unexpected vocabulary chunk.')


__all__ = [
    'Vocab', 'ExplicitVocab', 'FinalfusionBucketVocab', 'FastTextVocab',
    'SimpleVocab', 'load_finalfusion_bucket_vocab', 'load_fasttext_vocab',
    'load_explicit_vocab', 'load_simple_vocab', 'load_vocab'
]