"""
Methods to load Storage types from files.
"""

import ffp.io
from ffp.storage.quantized import QuantizedArray
from ffp.storage.ndarray import NdArray
from ffp.storage.storage import Storage


def load_storage(path: str, mmap=False) -> Storage:
    """
    Load Storage from the given finalfusion file
    :param path: Path of file in finalfusion format
    :param mmap: whether to mmap the storage
    :return: Storage
    """
    with open(path, "rb") as file:
        storage_chunks = [
            ffp.io.ChunkIdentifier.NdArray,
            ffp.io.ChunkIdentifier.QuantizedArray
        ]
        chunk = ffp.io.find_chunk(file, storage_chunks)
        if chunk == ffp.io.ChunkIdentifier.QuantizedArray:
            return QuantizedArray.load(file, mmap)
        if chunk == ffp.io.ChunkIdentifier.NdArray:
            return NdArray.load(file, mmap)
        raise ValueError("Can't find storage chunk")


def load_ndarray(path: str, mmap: bool = False) -> Storage:
    """
    Load an array chunk from the given file.
    :param path: File containing array chunk in finalfusion format
    :param mmap: whether to memory map the storage
    :return: NdArray
    """
    with open(path, "rb") as file:
        chunk = ffp.io.find_chunk(file, [ffp.io.ChunkIdentifier.NdArray])
        if chunk is None:
            raise ValueError("cannot find NdArray chunk")
        if chunk == ffp.io.ChunkIdentifier.NdArray:
            if mmap:
                return NdArray.mmap_chunk(file)
            return NdArray.read_chunk(file)
        raise ValueError("unknown storage type: " + str(chunk))


def load_quantized_array(path: str, mmap: bool = False) -> Storage:
    """
    Load an array chunk from the given file.
    :param path: File containing array chunk in finalfusion format
    :param mmap: whether to memory map the storage
    :return: NdArray
    """
    with open(path, "rb") as file:
        chunk = ffp.io.find_chunk(file,
                                  [ffp.io.ChunkIdentifier.QuantizedArray])
        if chunk == ffp.io.ChunkIdentifier.QuantizedArray:
            return QuantizedArray.load(file, mmap)
        raise ValueError("Can't find QuantizedArray.")


__all__ = ['load_ndarray', 'load_quantized_array', 'load_storage']
