import os
import tempfile

import ffp.io


def test_header_roundtrip():
    filename = os.path.join(tempfile.gettempdir(), "header.fifu")
    h = ffp.io.Header(
        [ffp.io.ChunkIdentifier.SimpleVocab, ffp.io.ChunkIdentifier.NdArray])
    with open(filename, 'wb') as f:
        h.write_chunk(f)

    with open(filename, 'rb') as f:
        h2 = ffp.io.Header.read_chunk(f)

    assert h.chunk_ids == h2.chunk_ids


def test_read_header(tests_root):
    filename = os.path.join(tests_root, "data", "simple_vocab.fifu")
    with open(filename, 'rb') as f:
        h = ffp.io.Header.read_chunk(f)
    assert h.chunk_ids == [
        ffp.io.ChunkIdentifier.SimpleVocab, ffp.io.ChunkIdentifier.NdArray
    ]
