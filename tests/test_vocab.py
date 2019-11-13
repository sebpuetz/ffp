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


def test_ff_buckets_constructor():
    v = ffp.vocab.FinalfusionBucketVocab([str(i) for i in range(10)])
    assert [v[str(i)] for i in range(10)] == [i for i in range(10)]
    with pytest.raises(ValueError):
        v = ffp.vocab.FinalfusionBucketVocab(["a"] * 2)
    assert len(v) == 10
    assert v.idx_bound == len(v) + pow(2, 21)


def test_fasttext_constructor():
    v = ffp.vocab.FastTextVocab([str(i) for i in range(10)])
    assert [v[str(i)] for i in range(10)] == [i for i in range(10)]
    with pytest.raises(ValueError):
        v = ffp.vocab.FastTextVocab(["a"] * 2)
    assert len(v) == 10
    assert v.idx_bound == len(v) + 2000000


def test_ff_buckets_roundtrip(tests_root):
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, "write_ff_buckets.fifu")
    v = ffp.vocab.Vocab.read(os.path.join(tests_root, "data", "ff_buckets.fifu"))
    v.write(filename)
    assert v == ffp.vocab.Vocab.read(filename)


def test_ngrams():
    assert ffp.vocab.word_ngrams("Test") == [
        "<Test>", "<Test", "<Tes", "<Te", "Test>", "Test", "Tes", "est>",
        "est", "st>"
    ]
    assert ffp.vocab.word_ngrams("Test",
                                 bracket=False) == ["Test", "Tes", "est"]
    assert ffp.vocab.word_ngrams("Test",
                                 min_n=5) == ["<Test>", "<Test", "Test>"]
    with pytest.raises(TypeError):
        _ = ffp.vocab.word_ngrams(None)
    with pytest.raises(TypeError):
        _ = ffp.vocab.word_ngrams(2)
    with pytest.raises(TypeError):
        _ = ffp.vocab.word_ngrams("Test", "not an int")
    with pytest.raises(ValueError):
        _ = ffp.vocab.word_ngrams("Test", 7)
    with pytest.raises(ValueError):
        _ = ffp.vocab.word_ngrams("Test", 0)
    with pytest.raises(ValueError):
        _ = ffp.vocab.word_ngrams("Test", 3, 0)
    with pytest.raises(ValueError):
        _ = ffp.vocab.word_ngrams("Test", 0, 0)


def test_subword_indices(tests_root):
    v = ffp.vocab.Vocab.read(
        os.path.join(tests_root, "data", "ff_buckets.fifu"))
    tuebingen_buckets = [
        14, 69, 74, 124, 168, 181, 197, 246, 250, 276, 300, 308, 325, 416, 549,
        590, 648, 651, 707, 717, 761, 817, 820, 857, 860, 1007
    ]
    assert sorted(v.subword_indices("tübingen", True)) == tuebingen_buckets
