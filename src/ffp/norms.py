"""
Norms module.
"""

import struct

import numpy as np

from ffp.io import Chunk, find_chunk, ChunkIdentifier, TypeId, _pad_float32


class Norms(np.ndarray, Chunk):
    """
    Norms subclass `numpy.ndarray`, all typical numpy operations are available.

    **Note:** Norms should be compatible with number of in-vocabulary embeddings.
    """
    def __new__(cls, array):
        assert array.ndim == 1, "norms need to be 1-d"
        if array.dtype != np.float32:
            raise TypeError("1-d float array expected")
        obj = array.view(cls)
        return obj

    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.NdNorms

    @staticmethod
    def read_chunk(file) -> 'Norms':
        n_norms, dtype = struct.unpack("<QI", file.read(struct.calcsize("QI")))
        assert TypeId(
            dtype) == TypeId.f32, "Expected f32 norms, found: " + str(
                TypeId(dtype))
        padding = _pad_float32(file.tell())
        file.seek(padding, 1)
        array = file.read(struct.calcsize("f") * n_norms)
        array = np.ndarray(buffer=array, shape=(n_norms, ), dtype=np.float32)
        return Norms(array)

    def write_chunk(self, file):
        file.write(struct.pack("<I", int(self.chunk_identifier())))
        padding = _pad_float32(file.tell())
        chunk_len = struct.calcsize(
            "QI") + padding + self.size * struct.calcsize("f")
        file.write(
            struct.pack("<QQI" + "x" * padding, chunk_len, self.size,
                        int(TypeId.f32)))
        file.write(self.tobytes())

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Norms(super().__getitem__(key))
        return super().__getitem__(key)


def load_norms(path: str):
    """
    Read a Norms chunk from the given finalfusion file.
    :param path: path
    """
    with open(path, "rb") as file:
        chunk = find_chunk(file, [ChunkIdentifier.NdNorms])
        if chunk is None:
            raise IOError("cannot find Norms chunk")
        if chunk == ChunkIdentifier.NdNorms:
            return Norms.read_chunk(file)
        raise IOError("unexpected chunk: " + str(chunk))
