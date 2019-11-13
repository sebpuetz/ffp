"""
finalfusion metadata
"""

import struct
from typing import IO

import toml

import ffp.io


class Metadata(dict, ffp.io.Chunk):
    """
    Embeddings metadata
    """
    @staticmethod
    def read(filename: str) -> 'Metadata':
        """
        Read a Metadata chunk from the given file.
        :param filename: filename
        """
        with open(filename, 'rb') as file:
            chunk = ffp.io.find_chunk(file, [ffp.io.ChunkIdentifier.Metadata])
            if chunk is None:
                raise IOError("cannot find Metadata chunk")
            if chunk == ffp.io.ChunkIdentifier.Metadata:
                return Metadata.read_chunk(file)
            raise IOError("unexpected chunk: " + str(chunk))

    @staticmethod
    def chunk_identifier():
        return ffp.io.ChunkIdentifier.Metadata

    @staticmethod
    def read_chunk(file: IO[bytes]) -> 'Metadata':
        file.seek(-12, 1)
        chunk_id, chunk_len = struct.unpack("<IQ",
                                            file.read(struct.calcsize("<IQ")))
        assert ffp.io.ChunkIdentifier(chunk_id) == Metadata.chunk_identifier()
        return Metadata(toml.loads(file.read(chunk_len).decode("utf-8")))

    def write_chunk(self, file: IO[bytes]):
        b_data = bytes(toml.dumps(self), "utf-8")
        file.write(
            struct.pack("<IQ", int(self.chunk_identifier()), len(b_data)))
        file.write(b_data)
