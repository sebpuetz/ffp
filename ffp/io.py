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
from typing import IO, Optional, Tuple, Iterable

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


@unique
class TypeId(IntEnum):
    """
    Enum identifying the different data types in finalfusion arrays.
    """
    u8 = 1
    f32 = 10


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

    def write(self, filename):
        """
        Write the chunk to the given filename in finalfusion format
        :param filename: filename
        """
        with open(filename, "wb") as file:
            chunk_id = self.chunk_identifier()
            if chunk_id == ChunkIdentifier.Header:
                raise ValueError("Cannot write header to file by itself")
            Header([chunk_id]).write_chunk(file)
            self.write_chunk(file)

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


def find_chunk(file: IO[bytes],
               chunks: Iterable[ChunkIdentifier]) -> Optional[ChunkIdentifier]:
    """
    Find one of the specified chunks in the given finalfusion file.

    Seeks the file to the beginning of the first chunk in `chunks` found.
    :param file: finalfusion file
    :param chunks: iterable of chunk identifiers
    :return: the first found chunk identifier
    """
    file.seek(0)
    Header.read_chunk(file)
    while True:
        chunk = Chunk.read_chunk_header(file)
        if chunk is None:
            return None
        if chunk[0] in chunks:
            return chunk[0]
        file.seek(chunk[1], 1)


def pad_float(pos):
    """
    Return the required padding to the next page boundary from a given position.
    :param pos:
    :return:
    """
    float_size = struct.calcsize('<f')
    return float_size - (pos % float_size)
