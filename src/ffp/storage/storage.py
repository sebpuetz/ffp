"""
Storage
"""

import abc
from typing import Tuple, BinaryIO


class Storage:
    """
    Common interface to finalfusion storage types.
    """
    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """
        The storage shape

        Returns
        -------
        (rows, cols) : Tuple[int, int]
            Tuple with storage dimensions
        """
    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, file: BinaryIO, mmap=False) -> 'Storage':
        """
        Load Storage from the given finalfusion file.

        Parameters
        ----------
        file : BinaryIO
            File at the beginning of a finalfusion storage
        mmap : bool
            Toggles memory mapping the buffer.

        Returns
        -------
        storage : Storage
            The storage from the file.
        """
    @staticmethod
    @abc.abstractmethod
    def mmap_storage(file: BinaryIO) -> 'Storage':
        """
        Memory map the storage.

        Parallel method to :func:`ffp.io.Chunk.read_chunk`. Instead of storing the
        :class:`Storage` in-memory, it memory maps the embeddings.

        Parameters
        ----------
        file : BinaryIO
            File at the beginning of a finalfusion storage

        Returns
        -------
        storage : Storage
            The memory mapped storage.
        """


__all__ = ['Storage']
