"""
Norms module.
"""

import struct

import numpy as np

from ffp.io import Chunk, find_chunk, ChunkIdentifier, TypeId, _pad_float32, _read_binary, \
    FinalfusionFormatError, _write_binary


class Norms(np.ndarray, Chunk):
    """
    Embedding Norms.

    Norms subclass `numpy.ndarray`, all typical numpy operations are available.

    The ith norm is expected to correspond to the l2 norm of the ith row in the storage before
    normalizing it. Therefore, Norms should have at most the same length as a given Storage and
    are expected to match the length of the Vocabulary.
    """
    def __new__(cls, array: np.ndarray):
        if array.dtype != np.float32 or array.ndim != 1:
            raise TypeError("expected 1-d float32 array")
        return array.view(cls)

    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.NdNorms

    @staticmethod
    def read_chunk(file) -> 'Norms':
        n_norms, type_id = _read_binary(file, "<QI")
        if int(TypeId.f32) != type_id:
            raise FinalfusionFormatError(
                f"Invalid Type, expected {TypeId.f32}, got {type_id}")
        padding = _pad_float32(file.tell())
        file.seek(padding, 1)
        array = np.fromfile(file=file, count=n_norms, dtype=np.float32)
        return Norms(array)

    def write_chunk(self, file):
        _write_binary(file, "<I", int(self.chunk_identifier()))
        padding = _pad_float32(file.tell())
        chunk_len = struct.calcsize(
            "QI") + padding + self.size * struct.calcsize("f")
        _write_binary(file, f"<QQI{padding}x", chunk_len, self.size,
                      int(TypeId.f32))
        self.tofile(file)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Norms(super().__getitem__(key))
        return super().__getitem__(key)


def load_norms(path: str):
    """
    Load an Norms chunk from the given file.

    Parameters
    ----------
    path : str
        Finalfusion file with a norms chunk.

    Returns
    -------
    storage : Norms
        The Norms from the file.

    Raises
    ------
    ValueError
        If the file did not contain an Norms chunk.
    """
    with open(path, "rb") as file:
        chunk = find_chunk(file, [ChunkIdentifier.NdNorms])
        if chunk is None:
            raise IOError("cannot find Norms chunk")
        if chunk == ChunkIdentifier.NdNorms:
            return Norms.read_chunk(file)
        raise IOError(f"unexpected chunk: {str(chunk)}")


__all__ = ['Norms', 'load_norms']
