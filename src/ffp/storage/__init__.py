"""
Finalfusion storage
"""

from ffp.storage.storage import Storage
from ffp.storage.ndarray import NdArray
from ffp.storage.quantized import QuantizedArray
from ffp.storage.io import load_storage, load_ndarray, load_quantized_array

__all__ = [
    'Storage', 'NdArray', 'QuantizedArray', 'load_storage',
    'load_quantized_array', 'load_ndarray'
]
