"""
Finalfusion storage
"""
from ffp.io import find_chunk, ChunkIdentifier

from ffp.storage.storage import Storage
from ffp.storage.ndarray import NdArray, load_ndarray
from ffp.storage.quantized import QuantizedArray, load_quantized_array


def load_storage(path: str, mmap: bool = False) -> Storage:
    """
    Load any storage from a finalfusion file.

    Loads the first known storage from a finalfusion file.

    Parameters
    ----------
    path : str
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
    with open(path, "rb") as file:
        chunk = find_chunk(
            file, [ChunkIdentifier.NdArray, ChunkIdentifier.QuantizedArray])
        if chunk is None:
            raise ValueError('File did not contain a storage')
        if chunk == ChunkIdentifier.NdArray:
            return NdArray.load(file, mmap)
        if chunk == ChunkIdentifier.QuantizedArray:
            return QuantizedArray.load(file, mmap)
        raise ValueError('Unexpected storage chunk.')


__all__ = [
    'Storage', 'NdArray', 'QuantizedArray', 'load_storage',
    'load_quantized_array', 'load_ndarray'
]
