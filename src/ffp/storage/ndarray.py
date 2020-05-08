"""
Finalfusion NdArray Storage
"""

import struct
from os import PathLike
from typing import Tuple, BinaryIO, Union

import numpy as np

from ffp.io import ChunkIdentifier, TypeId, FinalfusionFormatError, _pad_float32, _read_binary, \
    _write_binary, find_chunk, Chunk
from ffp.storage.storage import Storage


class NdArray(np.ndarray, Chunk, Storage):
    """
    NdArray(array: numpy.ndarray)

    Array storage.

    Essentially a numpy matrix, either in-memory or memory-mapped.

    Examples
    --------
    >>> matrix = np.float32(np.random.rand(10, 50))
    >>> ndarray_storage = NdArray(matrix)
    >>> np.allclose(matrix, ndarray_storage)
    True
    >>> ndarray_storage.shape
    (10, 50)
    """
    def __new__(cls, array: np.ndarray):
        """
        Construct a new NdArray storage.

        Parameters
        ----------
        array : numpy.ndarray
            The storage buffer.

        Raises
        ------
        TypeError
            If the array is not a 2-dimensional float32 array.
        """
        if array.dtype != np.float32 or array.ndim != 2:
            raise TypeError("expected 2-d float32 array")
        return array.view(cls)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        The storage shape

        Returns
        -------
        (rows, cols) : Tuple[int, int]
            Tuple with storage dimensions
        """
        return super().shape

    @classmethod
    def load(cls, file: BinaryIO, mmap=False) -> 'NdArray':
        return cls.mmap_storage(file) if mmap else cls.read_chunk(file)

    @staticmethod
    def read_chunk(file: BinaryIO) -> 'NdArray':
        rows, cols = NdArray._read_array_header(file)
        array = np.fromfile(file=file, count=rows * cols, dtype=np.float32)
        array = np.reshape(array, (rows, cols))
        return NdArray(array)

    @staticmethod
    def mmap_storage(file: BinaryIO) -> 'NdArray':
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
    def chunk_identifier():
        return ChunkIdentifier.NdArray

    @staticmethod
    def _read_array_header(file: BinaryIO) -> Tuple[int, int]:
        """
        Helper method to read the header of an NdArray chunk.

        The method reads the shape tuple, verifies the TypeId and seeks the file to the start
        of the array. The shape tuple is returned.

        Parameters
        ----------
        file : BinaryIO
            finalfusion file with a storage at the start of a NdArray chunk.

        Returns
        -------
        shape : Tuple[int, int]
            Shape of the storage.

        Raises
        ------
        FinalfusionFormatError
            If the TypeId does not match TypeId.f32
        """
        rows, cols = _read_binary(file, "<QI")
        type_id = _read_binary(file, "<I")[0]
        if int(TypeId.f32) != type_id:
            raise FinalfusionFormatError(
                f"Invalid Type, expected {TypeId.f32}, got {type_id}")
        file.seek(_pad_float32(file.tell()), 1)
        return rows, cols

    def write_chunk(self, file: BinaryIO):
        _write_binary(file, "<I", int(self.chunk_identifier()))
        padding = _pad_float32(file.tell())
        chunk_len = struct.calcsize("<QII") + padding + struct.calcsize(
            f'<{self.size}f')
        # pylint: disable=unpacking-non-sequence
        rows, cols = self.shape
        _write_binary(file, "<QQII", chunk_len, rows, cols, int(TypeId.f32))
        _write_binary(file, f"{padding}x")
        self.tofile(file)

    def __getitem__(self, key) -> Union[np.ndarray, 'NdArray']:
        if isinstance(key, slice):
            return super().__getitem__(key)
        return super().__getitem__(key).view(np.ndarray)


def load_ndarray(file: Union[str, bytes, int, PathLike],
                 mmap: bool = False) -> NdArray:
    """
    Load an array chunk from the given file.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Finalfusion file with a ndarray chunk.
    mmap : bool
        Toggles memory mapping the array buffer as read only.

    Returns
    -------
    storage : NdArray
        The NdArray storage from the file.

    Raises
    ------
    ValueError
        If the file did not contain an NdArray chunk.
    """
    with open(file, "rb") as inf:
        chunk = find_chunk(inf, [ChunkIdentifier.NdArray])
        if chunk is None:
            raise ValueError("File did not contain a NdArray chunk")
        if chunk == ChunkIdentifier.NdArray:
            if mmap:
                return NdArray.mmap_storage(inf)
            return NdArray.read_chunk(inf)
        raise ValueError(f"unknown storage type: {chunk}")


__all__ = ['NdArray', 'load_ndarray']
