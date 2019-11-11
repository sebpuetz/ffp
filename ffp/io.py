"""
Define some common IO operations and types.

`Chunk`s are the basic blocks of finalfusion embeddings, each component is serialized
as a `Chunk` in finalfusion files, starting with a `ChunkIdentifier` followed by the
chunk size in bytes. This design allows straight-forward reading of the file and
skipping unwanted parts.
"""
import struct
from abc import ABC, abstractmethod
from enum import unique, IntEnum
from typing import IO, Optional, Tuple

MAGIC = b'FiFu'
VERSION = 0


@unique
class ChunkIdentifier(IntEnum):
    """
    Enum identifying the different `Chunk` types.
    """
    Header = 0
    SimpleVocab = 1
    NdArray = 2
    BucketSubwordVocab = 3
    QuantizedArray = 4
    Metadata = 5
    NdNorms = 6
    FastTextSubwordVocab = 7
    ExplicitSubwordVocab = 8


class Chunk(ABC):
    """
    Common methods that all finalfusion `Chunk`s need to implement.
    """
    @staticmethod
    def read_chunk_header(file) -> Optional[Tuple[ChunkIdentifier, int]]:
        """
        Reads the chunk header, after successfully reading the header, `ChunkIdentifier` and
        the chunk size in bytes are returned.

        :param file: file in finalfusion format at the beginning of a chunk.
        :return: (ChunkIdentifier, chunk_size)
        """
        buffer = file.read(12)
        if len(buffer) < 12:
            return None
        chunk_id, chunk_size = struct.unpack("<IQ", buffer)
        return ChunkIdentifier(chunk_id), chunk_size

    @staticmethod
    @abstractmethod
    def chunk_identifier() -> ChunkIdentifier:
        """
        Get this `Chunk`'s identifier
        :return: ChunkIdentifier
        """
    @staticmethod
    @abstractmethod
    def read_chunk(file: IO[bytes]) -> 'Chunk':
        """
        Read a chunk and return self.
        :param file: File at the beginning of a chunk.
        :return: self
        """
    @abstractmethod
    def write_chunk(self, file: IO[bytes]):
        """
        Append the chunk to a file.
        :param file: the file to append the chunk to.
        """


class Header(Chunk):
    """
    Header Chunk

    The header chunk handles the preamble.
    """
    def __init__(self, chunk_ids):
        self.chunk_ids_ = chunk_ids

    @property
    def chunk_ids(self) -> list:
        """
        Get the chunk IDs from the header
        :return: Chunk identifiers
        """
        return self.chunk_ids_

    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.Header

    @staticmethod
    def read_chunk(file):
        magic = file.read(4)
        if magic != MAGIC:
            raise IOError("Magic should be b'FiFu', not: " +
                          magic.decode('utf-8'))
        version = struct.unpack("<I", file.read(4))[0]
        if version != VERSION:
            raise IOError("Unknown model version: " + version)
        n_chunks = struct.unpack("<I", file.read(4))[0]
        chunk_ids = list(
            struct.unpack("<" + "I" * n_chunks, file.read(4 * n_chunks)))
        return Header(chunk_ids)

    def write_chunk(self, file):
        file.write(MAGIC)
        n_chunks = len(self.chunk_ids)
        file.write(
            struct.pack("<II" + "I" * n_chunks, VERSION, n_chunks,
                        *self.chunk_ids))
