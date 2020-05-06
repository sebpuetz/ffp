"""
finalfusion metadata
"""

import struct
from typing import IO

import toml

from ffp.io import Chunk, ChunkIdentifier, find_chunk


class Metadata(dict, Chunk):
    """
    Embeddings metadata
    """
    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.Metadata

    @staticmethod
    def read_chunk(file: IO[bytes]) -> 'Metadata':
        file.seek(-12, 1)
        chunk_id, chunk_len = struct.unpack("<IQ",
                                            file.read(struct.calcsize("<IQ")))
        assert ChunkIdentifier(chunk_id) == Metadata.chunk_identifier()
        return Metadata(toml.loads(file.read(chunk_len).decode("utf-8")))

    def write_chunk(self, file: IO[bytes]):
        b_data = bytes(toml.dumps(self), "utf-8")
        file.write(
            struct.pack("<IQ", int(self.chunk_identifier()), len(b_data)))
        file.write(b_data)


def load_metadata(path: str) -> Metadata:
    """
    Read a Metadata chunk from the given file.
    :param path: filename
    """
    with open(path, 'rb') as file:
        chunk = find_chunk(file, [ChunkIdentifier.Metadata])
        if chunk is None:
            raise IOError("cannot find Metadata chunk")
        if chunk == ChunkIdentifier.Metadata:
            return Metadata.read_chunk(file)
        raise IOError("unexpected chunk: " + str(chunk))


__all__ = ['Metadata', 'load_metadata']
