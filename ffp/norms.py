"""
Norms module.
"""

import struct

import numpy as np

import ffp.io


class Norms(np.ndarray, ffp.io.Chunk):
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
    def read(filename: str) -> 'Norms':
        """
        Read a Norms chunk from the given file.
        :param filename: filename
        """
        with open(filename, "rb") as file:
            chunk = ffp.io.find_chunk(file, [ffp.io.ChunkIdentifier.NdNorms])
            if chunk is None:
                raise IOError("cannot find Norms chunk")
            if chunk == ffp.io.ChunkIdentifier.NdNorms:
                return Norms.read_chunk(file)
            raise IOError("unexpected chunk: " + str(chunk))

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.NdNorms

    @staticmethod
    def read_chunk(file) -> 'Norms':
        n_norms, dtype = struct.unpack("<QI", file.read(struct.calcsize("QI")))
        assert ffp.io.TypeId(
            dtype) == ffp.io.TypeId.f32, "Expected f32 norms, found: " + str(
                ffp.io.TypeId(dtype))
        padding = ffp.io.pad_float(file.tell())
        file.seek(padding, 1)
        array = file.read(struct.calcsize("f") * n_norms)
        array = np.ndarray(buffer=array, shape=(n_norms, ), dtype=np.float32)
        return Norms(array)

    def write_chunk(self, file):
        file.write(struct.pack("<I", int(self.chunk_identifier())))
        padding = ffp.io.pad_float(file.tell())
        chunk_len = struct.calcsize(
            "QI") + padding + self.size * struct.calcsize("f")
        file.write(
            struct.pack("<QQI" + "x" * padding, chunk_len, self.size,
                        int(ffp.io.TypeId.f32)))
        file.write(self.tobytes())

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Norms(super().__getitem__(key))
        return super().__getitem__(key)
