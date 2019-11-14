import os
import tempfile

import numpy as np
import pytest
import ffp

TEST_NORMS = [
    6.557438373565674, 8.83176040649414, 6.164413928985596, 9.165151596069336,
    7.4833149909973145, 7.211102485656738, 7.4833149909973145
]


def test_read_embeddings(tests_root):
    ffp.embeddings.Embeddings.read(
        os.path.join(tests_root, "data", "simple_vocab.fifu"))
    e = ffp.embeddings.Embeddings.read(
        os.path.join(tests_root, "data", "ff_buckets.fifu"))
    e2 = ffp.embeddings.Embeddings.read(
        os.path.join(tests_root, "data", "ff_buckets.fifu"),
        "finalfusion_mmap")
    assert np.allclose(e.storage, e2.storage)
    with pytest.raises(TypeError):
        ffp.embeddings.Embeddings.read(None)
    with pytest.raises(IOError):
        ffp.embeddings.Embeddings.read(1)
    with pytest.raises(IOError):
        ffp.embeddings.Embeddings.read("foo")


def test_embeddings_from_storage():
    matrix = np.tile(np.arange(0, 10, dtype=np.float32), (10, 1))
    s = ffp.storage.NdArray(matrix)
    e = ffp.embeddings.Embeddings(storage=s)
    assert np.allclose(e.storage, matrix)
    assert np.allclose(s, matrix)


def test_embeddings_from_matrix():
    matrix = np.tile(np.arange(0, 10, dtype=np.float32), (10, 1))
    e = ffp.embeddings.Embeddings(storage=matrix)
    assert np.allclose(e.storage, matrix)


def test_ff_embeddings_roundtrip(embeddings_fifu, vocab_array_tuple):
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, "write_embeddings.fifu")
    v = embeddings_fifu.vocab
    s = embeddings_fifu.storage
    embeddings_fifu.write(filename)
    assert v.words == vocab_array_tuple[0]
    matrix = vocab_array_tuple[1]
    matrix = matrix.squeeze() / np.linalg.norm(matrix, axis=1, keepdims=True)
    assert np.allclose(matrix, s)
    assert np.allclose(s, ffp.embeddings.Embeddings.read(filename).storage)


def test_ff_embeddings_roundtrip_ff_buckets(bucket_vocab_embeddings_fifu):
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, "write_embeddings.fifu")
    bucket_vocab_embeddings_fifu.write(filename)
    e2 = ffp.embeddings.Embeddings.read(filename)
    assert bucket_vocab_embeddings_fifu.vocab == e2.vocab
    assert bucket_vocab_embeddings_fifu.metadata == e2.metadata
    assert np.allclose(bucket_vocab_embeddings_fifu.storage, e2.storage)
    assert np.allclose(bucket_vocab_embeddings_fifu.norms, e2.norms)


def test_embeddings_lookup(embeddings_fifu, vocab_array_tuple):
    matrix = vocab_array_tuple[1]
    matrix = matrix.squeeze() / np.linalg.norm(matrix, axis=1, keepdims=True)
    for i, word in enumerate(vocab_array_tuple[0]):
        assert np.allclose(embeddings_fifu[word], matrix[i])
    lookup = np.zeros_like(matrix)
    for i, word in enumerate(vocab_array_tuple[0]):
        embeddings_fifu.embedding(word, out=lookup[i])
        emb = embeddings_fifu.embedding(word)
        assert np.allclose(emb, lookup[i])
    assert np.allclose(matrix, lookup)
    with pytest.raises(KeyError):
        _ = embeddings_fifu["foo"]
    with pytest.raises(KeyError):
        _ = embeddings_fifu[None]
    with pytest.raises(KeyError):
        _ = embeddings_fifu[1]


def test_unknown_embeddings(embeddings_fifu):
    assert embeddings_fifu.embedding(
        "OOV") is None, "Unknown lookup with no default failed"
    assert embeddings_fifu.embedding(
        "OOV",
        default=None) is None, "Unknown lookup with 'None' default failed"
    assert np.allclose(
        embeddings_fifu.embedding("OOV",
                                  default=np.zeros(10, dtype=np.float32)),
        np.array([0.] * 10)), "Unknown lookup with 'list' default failed"


def test_embeddings_with_norms_oov(embeddings_fifu):
    assert embeddings_fifu.embedding_with_norm(
        "Something out of vocabulary") is None


def test_indexing(embeddings_fifu):
    assert embeddings_fifu["one"] is not None
    with pytest.raises(KeyError):
        embeddings_fifu["Something out of vocabulary"]


def test_embeddings_oov(embeddings_fifu):
    assert embeddings_fifu.embedding("Something out of vocabulary") is None


def test_norms(embeddings_fifu):
    for i, (embedding, norm) in enumerate(zip(embeddings_fifu, TEST_NORMS)):
        assert pytest.approx(embedding[2]) == norm, "Norm fails to match!"
        w = embeddings_fifu.vocab.words[i]
        assert embeddings_fifu.embedding_with_norm(w)[1] == norm
        emb_with_norm = embeddings_fifu.embedding_with_norm(w)
        assert emb_with_norm[1] == norm
        out = np.zeros_like(emb_with_norm[0])
        emb_with_out = embeddings_fifu.embedding_with_norm(w, out=out)
        assert np.allclose(emb_with_norm[0], emb_with_out[0])
        assert norm == emb_with_out[1]


def test_no_norms(vocab_array_tuple):
    vocab, matrix = vocab_array_tuple
    embeddings = ffp.embeddings.Embeddings(vocab=ffp.vocab.SimpleVocab(vocab),
                                           storage=matrix)
    with pytest.raises(TypeError):
        _ = embeddings.embedding_with_norm("bla")


def test_embeddings(embeddings_fifu, embeddings_text, embeddings_text_dims,
                    embeddings_w2v):
    assert len(embeddings_fifu.vocab) == 7
    assert len(embeddings_text.vocab) == 7
    assert len(embeddings_text_dims.vocab) == 7
    assert len(embeddings_w2v.vocab) == 7
    fifu_storage = embeddings_fifu.storage
    assert fifu_storage.shape == (7, 10)

    for embedding, storage_row in zip(embeddings_fifu, fifu_storage):
        assert np.allclose(
            embedding[1],
            embeddings_text[embedding[0]]), "FiFu and text embedding mismatch"
        assert np.allclose(embedding[1], embeddings_text_dims[
            embedding[0]]), "FiFu and textdims embedding mismatch"
        assert np.allclose(
            embedding[1],
            embeddings_w2v[embedding[0]]), "FiFu and w2v embedding mismatch"
        assert np.allclose(embedding[1],
                           storage_row), "FiFu and storage row  mismatch"
