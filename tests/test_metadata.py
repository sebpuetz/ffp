import os
import tempfile

import pytest

import ffp.metadata


def test_read_metadata(tests_root):
    with pytest.raises(TypeError):
        ffp.metadata.Metadata.read(None)
    with pytest.raises(IOError):
        ffp.metadata.Metadata.read(1)
    with pytest.raises(IOError):
        ffp.metadata.Metadata.read("foo")
    m = ffp.metadata.Metadata.read(
        os.path.join(tests_root, "data", "ff_buckets.fifu"))
    assert "common_config" in m
    assert m["common_config"]["dims"] == 5
    assert m["vocab_config"]["type"] == "SubwordVocab"


def test_metadata_roundtrip(tests_root):
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, "write_meta.fifu")
    m = ffp.metadata.Metadata.read(
        os.path.join(tests_root, "data", "ff_buckets.fifu"))
    m.write(filename)
    m2 = ffp.metadata.Metadata.read(filename)
    assert m == m2


def test_metadata_dict():
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, "write_meta.fifu")
    metadata = ffp.metadata.Metadata({"test": 1, "test2": "test"})
    metadata.write(filename)
    m2 = ffp.metadata.Metadata.read(filename)
    assert metadata == m2


def test_metadata_invalid_key():
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, "write_meta.fifu")
    metadata = ffp.metadata.Metadata({1: "test", "test2": "test"})
    with pytest.raises(KeyError):
        metadata.write(filename)
