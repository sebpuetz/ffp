"""
finalfusion metadata
"""
from os import PathLike
from typing import BinaryIO, Union

import toml

from ffp.io import Chunk, ChunkIdentifier, find_chunk, _read_binary, _write_binary


class Metadata(dict, Chunk):
    """
    Embeddings metadata

    Metadata can be used as a regular Python dict. For serialization, the contents need to be
    serializable through `toml.dumps`. Finalfusion assumes metadata to be a TOML formatted
    string.
    """
    @staticmethod
    def chunk_identifier():
        return ChunkIdentifier.Metadata

    @staticmethod
    def read_chunk(file: BinaryIO) -> 'Metadata':
        file.seek(-12, 1)
        chunk_id, chunk_len = _read_binary(file, "<IQ")
        assert ChunkIdentifier(chunk_id) == Metadata.chunk_identifier()
        return Metadata(toml.loads(file.read(chunk_len).decode("utf-8")))

    def write_chunk(self, file: BinaryIO):
        b_data = bytes(toml.dumps(self), "utf-8")
        _write_binary(file, "<IQ", int(self.chunk_identifier()), len(b_data))
        file.write(b_data)


def load_metadata(path: Union[str, bytes, int, PathLike]) -> Metadata:
    """
    Load a Metadata chunk from the given file.

    Parameters
    ----------
    path : str
        Finalfusion file with a metadata chunk.

    Returns
    -------
    metadata : Metadata
        The Norms from the file.

    Raises
    ------
    ValueError
        If the file did not contain an Metadata chunk.
    """
    with open(path, 'rb') as file:
        chunk = find_chunk(file, [ChunkIdentifier.Metadata])
        if chunk is None:
            raise ValueError("File did not contain a Metadata chunk")
        if chunk == ChunkIdentifier.Metadata:
            return Metadata.read_chunk(file)
        raise ValueError(f"unexpected chunk: {str(chunk)}" + str(chunk))


__all__ = ['Metadata', 'load_metadata']
