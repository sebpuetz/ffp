import os

import pytest
import ffp.vocab
import tempfile


def test_reading(tests_root):
    with pytest.raises(TypeError):
        ffp.vocab.Vocab.read(None)
    with pytest.raises(IOError):
        ffp.vocab.Vocab.read(1)
    with pytest.raises(IOError):
        ffp.vocab.Vocab.read("foo")
    v = ffp.vocab.Vocab.read(
        os.path.join(tests_root, "data", "simple_vocab.fifu"))
    assert v.words[0] == "Paris"


def test_contains():
    v = ffp.vocab.SimpleVocab([str(i) for i in range(10)])
    assert "1" in v
    assert None not in v
    assert ["1", "2"] in v


def test_simple_roundtrip(tests_root):
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, "write_simple.fifu")
    v = ffp.vocab.Vocab.read(
        os.path.join(tests_root, "data", "simple_vocab.fifu"))
    v.write(filename)
    assert v == ffp.vocab.Vocab.read(filename)


def test_simple_constructor():
    v = ffp.vocab.SimpleVocab([str(i) for i in range(10)])
    assert [v[str(i)] for i in range(10)] == [i for i in range(10)]
    with pytest.raises(ValueError):
        ffp.vocab.SimpleVocab(["a"] * 2)
    assert len(v) == 10
    assert v.idx_bound == len(v)


def test_simple_eq():
    v = ffp.vocab.SimpleVocab([str(i) for i in range(10)])
    assert v == v
    with pytest.raises(TypeError):
        _ = v > v
    with pytest.raises(TypeError):
        _ = v >= v
    with pytest.raises(TypeError):
        _ = v <= v
    with pytest.raises(TypeError):
        _ = v < v
    v2 = ffp.vocab.SimpleVocab([str(i + 1) for i in range(10)])
    assert v != v2
    assert v in v


def test_string_idx(simple_vocab_fifu):
    assert simple_vocab_fifu["Paris"] == 0


def test_string_oov(simple_vocab_fifu):
    with pytest.raises(KeyError):
        simple_vocab_fifu["definitely in vocab"]
