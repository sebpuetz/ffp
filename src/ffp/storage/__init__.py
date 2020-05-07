"""
Finalfusion storage
"""
from os import PathLike
from typing import Union

from ffp.io import find_chunk, ChunkIdentifier

from ffp.storage.storage import Storage
from ffp.storage.ndarray import NdArray, load_ndarray
from ffp.storage.quantized import QuantizedArray, load_quantized_array


def load_storage(file: Union[str, bytes, int, PathLike],
                 mmap: bool = False) -> Storage:
    """
    Load any storage from a finalfusion file.

    Loads the first known storage from a finalfusion file.

    Parameters
    ----------
    file : str
        Path to file containing a finalfusion storage chunk.
    mmap : bool
        Toggles memory mapping the storage buffer as read-only.

    Returns
    -------
    vocab : Union[NdArray, QuantizedArray]
        First storage in the file.

    Raises
    ------
    ValueError
         If the file did not contain a storage.
    """
    with open(file, "rb") as inf:
        chunk = find_chunk(
            inf, [ChunkIdentifier.NdArray, ChunkIdentifier.QuantizedArray])
        if chunk is None:
            raise ValueError('File did not contain a storage')
        if chunk == ChunkIdentifier.NdArray:
            return NdArray.load(inf, mmap)
        if chunk == ChunkIdentifier.QuantizedArray:
            return QuantizedArray.load(inf, mmap)
        raise ValueError('Unexpected storage chunk.')


__all__ = [
    'Storage', 'NdArray', 'QuantizedArray', 'load_storage',
    'load_quantized_array', 'load_ndarray'
]
