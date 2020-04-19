"""
Storage
"""

import abc
from typing import Tuple, IO

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

    @classmethod
    def load(cls, file: IO[bytes], mmap=False) -> 'Storage':
        """
        Load Storage from the given finalfusion file.

        :param file: file object with finalfusion storage
        :param mmap: whether to mmap the storage
        :return: Storage
        """
        return cls.mmap_chunk(file) if mmap else cls.read_chunk(file)

    @staticmethod
    @abc.abstractmethod
    def mmap_chunk(file: IO[bytes]) -> 'Storage':
        """

        Memory maps the storage as a read-only buffer.
        :param file: File in finalfusion format containing a storage chunk.
        :return: Storage
        """
