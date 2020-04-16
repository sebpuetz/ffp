"""
Finalfusion storage
"""

import abc
import struct
from typing import IO, Tuple, List

import numpy as np

import ffp.io


class Storage(ffp.io.Chunk):
    """
    Common interface to finalfusion storage types.
    """
    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """
        Get the shape of the storage
        :return: int tuple containing (rows, cols)
        """
    @abc.abstractmethod
    def __getitem__(self, key):
        pass


class NdArray(np.ndarray, Storage):
    """
    Array storage.

    Wraps an numpy matrix, either in-memory or memory-mapped.
    """
    def __new__(cls, array: np.ndarray):
        if array.dtype != np.float32 or array.ndim != 2:
            raise TypeError("expected 2-d float32 array")
        obj = array.view(cls)
        return obj

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.NdArray

    @staticmethod
    def read_chunk(file) -> 'NdArray':
        rows, cols = NdArray._read_array_header(file)
        array = np.fromfile(file=file, count=rows * cols, dtype=np.float32)
        array = np.reshape(array, (rows, cols))
        return NdArray(array)

    @property
    def shape(self) -> Tuple[int, int]:
        return super().shape

    @staticmethod
    def mmap_chunk(file) -> 'NdArray':
        """
        Mmaps the storage as read-only. This method positions `file` after the last element of the
        buffer.

        :param file: file containing the storage, positioned at the first element of the buffer.
        :return: NdArray backed by the mmapped array
        """
        rows, cols = NdArray._read_array_header(file)
        offset = file.tell()
        file.seek(rows * cols * struct.calcsize('f'), 1)
        return NdArray(
            np.memmap(file.name,
                      dtype=np.float32,
                      mode='r',
                      offset=offset,
                      shape=(rows, cols)))

    @staticmethod
    def _read_array_header(file: IO[bytes]) -> Tuple[int, int]:
        """
        Read the header of an array chunk. This contains the matrix dimensions and the datatype.

        In addition, this method seeks the file to the beginning of the actual data.
        :param file:
        :return: tuple containing rows, cols
        """
        rows, cols = struct.unpack("<QI", file.read(struct.calcsize("<QI")))
        type_id = ffp.io.TypeId(
            struct.unpack("<I", file.read(struct.calcsize("<I")))[0])
        assert type_id == ffp.io.TypeId.f32, "Expected " + str(
            ffp.io.TypeId.f32) + ", found: " + str(type_id)
        file.seek(ffp.io.pad_float(file.tell()), 1)
        return rows, cols

    def write_chunk(self, file: IO[bytes]):
        file.write(struct.pack("<I", int(self.chunk_identifier())))
        padding = ffp.io.pad_float(file.tell())
        chunk_len = struct.calcsize(
            "<QII") + padding + self.size * struct.calcsize('<f')
        # pylint: disable=unpacking-non-sequence
        rows, cols = self.shape
        file.write(struct.pack("<QQI", chunk_len, rows, cols))
        file.write(struct.pack("<I", int(ffp.io.TypeId.f32)))
        file.write(struct.pack("x" * padding))
        file.write(self.tobytes())

    def __getitem__(self, key):
        if isinstance(key, slice):
            return super().__getitem__(key)
        return super().__getitem__(key).view(np.ndarray)


def load_storage(path: str, mmap=False) -> Storage:
    """
    Load Storage from the given finalfusion file
    :param path: Path of file in finalfusion format
    :param mmap: whether to mmap the storage
    :return: Storage
    """
    storage_chunks = [
        ffp.io.ChunkIdentifier.NdArray, ffp.io.ChunkIdentifier.QuantizedArray
    ]
    storage = _load(path, mmap, storage_chunks)
    if storage is None:
        raise ValueError("cannot find storage chunk")
    return storage


def load_ndarray(path: str, mmap: bool = False) -> 'NdArray':
    """
    Load an array chunk from the given file.
    :param file: File containing array chunk in finalfusion format
    :param mmap: whether to memory map the storage
    :return: NdArray
    """
    storage = _load(path, mmap, [ffp.io.ChunkIdentifier.NdArray])
    if storage is None:
        raise ValueError("cannot find NdArray chunk")
    return storage


def _load(path: str, mmap: bool, targets: List[ffp.io.ChunkIdentifier]):
    """
    Read the first storae chunk specified in `target` from `filename`.
    :param path: filename
    :param targets: List of target chunks
    :return: Storage
    """
    with open(path, "rb") as file:
        chunk = ffp.io.find_chunk(file, targets)
        if chunk is None:
            return None
        if chunk == ffp.io.ChunkIdentifier.NdArray:
            if mmap:
                return NdArray.mmap_chunk(file)
            return NdArray.read_chunk(file)
        if chunk == ffp.io.ChunkIdentifier.QuantizedArray:
            raise NotImplementedError(
                "Quantized storage is not yet implemented.")
        raise ValueError("unknown storage type: " + str(chunk))
