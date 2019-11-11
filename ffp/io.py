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
