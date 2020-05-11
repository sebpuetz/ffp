import os
import tempfile

import numpy as np
import pytest
import ffp
import ffp.io

TEST_NORMS = [
    6.557438373565674, 8.83176040649414, 6.164413928985596, 9.165151596069336,
    7.4833149909973145, 7.211102485656738, 7.4833149909973145
]


def test_read_embeddings(tests_root):
    ffp.load_finalfusion(os.path.join(tests_root, "data", "simple_vocab.fifu"))
    e = ffp.load_finalfusion(
        os.path.join(tests_root, "data", "ff_buckets.fifu"))
    e2 = ffp.load_finalfusion(os.path.join(tests_root, "data",
                                           "ff_buckets.fifu"),
                              mmap=True)
    assert np.allclose(e.storage, e2.storage)
    with pytest.raises(TypeError):
        ffp.load_finalfusion(None)
    with pytest.raises(ffp.io.FinalfusionFormatError):
        ffp.load_finalfusion(1)
    with pytest.raises(IOError):
        ffp.load_finalfusion("foo")


def test_embeddings_from_storage():
    matrix = np.tile(np.arange(0, 10, dtype=np.float32), (10, 1))
    s = ffp.storage.NdArray(matrix)
    e = ffp.Embeddings(storage=s)
    assert np.allclose(e.storage, matrix)
    assert np.allclose(s, matrix)


def test_set_norms(embeddings_fifu):
    n = ffp.norms.Norms(np.ones(len(embeddings_fifu.vocab), dtype=np.float32))
    embeddings_fifu.norms = n
    assert np.allclose(n, embeddings_fifu.norms)
    embeddings_fifu.norms = None
    assert embeddings_fifu.norms is None
    with pytest.raises(TypeError):
        embeddings_fifu.norms = "bla"
    with pytest.raises(TypeError):
        embeddings_fifu.norms = np.ones(len(embeddings_fifu.vocab),
                                        dtype=np.float32)
    with pytest.raises(AssertionError):
        embeddings_fifu.norms = ffp.norms.Norms(
            np.ones(len(embeddings_fifu.vocab) - 1, dtype=np.float32))
    with pytest.raises(AssertionError):
        embeddings_fifu.norms = ffp.norms.Norms(
            np.ones(len(embeddings_fifu.vocab) + 1, dtype=np.float32))
    assert embeddings_fifu.norms is None


def test_set_storage(embeddings_fifu):
    s = ffp.storage.NdArray(np.zeros_like(embeddings_fifu.storage))
    embeddings_fifu.storage = s
    assert np.allclose(embeddings_fifu.storage, s)
    embeddings_fifu.storage = None
    assert embeddings_fifu.storage is None
    s = ffp.storage.NdArray(
        np.ones((len(embeddings_fifu.vocab), 1), dtype=np.float32))
    embeddings_fifu.storage = s
    assert np.allclose(embeddings_fifu.storage, s)
    with pytest.raises(TypeError):
        embeddings_fifu.storage = "bla"
    with pytest.raises(TypeError):
        embeddings_fifu.storage = np.ones((len(embeddings_fifu.vocab), 1),
                                          dtype=np.float32)
    with pytest.raises(AssertionError):
        embeddings_fifu.storage = ffp.storage.NdArray(
            np.ones((len(embeddings_fifu.vocab) - 1, 1), dtype=np.float32))
    with pytest.raises(AssertionError):
        embeddings_fifu.storage = ffp.storage.NdArray(
            np.ones((len(embeddings_fifu.vocab) + 1, 1), dtype=np.float32))
    assert np.allclose(embeddings_fifu.storage, s)


def test_set_vocab(embeddings_fifu):
    v = ffp.vocab.SimpleVocab(
        [str(i) for i in range(len(embeddings_fifu.storage))])
    embeddings_fifu.vocab = v
    assert embeddings_fifu.vocab == v
    embeddings_fifu.vocab = None
    assert embeddings_fifu.vocab is None
    with pytest.raises(TypeError):
        embeddings_fifu.vocab = "bla"
    with pytest.raises(TypeError):
        embeddings_fifu.vocab = [
            str(i) for i in range(len(embeddings_fifu.storage))
        ]
    with pytest.raises(AssertionError):
        embeddings_fifu.vocab = ffp.vocab.SimpleVocab(
            [str(i) for i in range(len(embeddings_fifu.storage) - 1)])
    with pytest.raises(AssertionError):
        embeddings_fifu.vocab = ffp.vocab.SimpleVocab(
            [str(i) for i in range(len(embeddings_fifu.storage) + 1)])
    assert embeddings_fifu.vocab is None


def test_set_metadata(embeddings_fifu):
    m = ffp.metadata.Metadata({"test": "foo", "test2": 2})
    embeddings_fifu.metadata = m
    assert embeddings_fifu.metadata == m
    embeddings_fifu.metadata = None
    assert embeddings_fifu.metadata is None
    with pytest.raises(TypeError):
        embeddings_fifu.metadata = {}
    with pytest.raises(TypeError):
        embeddings_fifu.metadata = "m"
    assert embeddings_fifu.metadata is None


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
    assert np.allclose(s, ffp.load_finalfusion(filename).storage)


def test_ff_embeddings_roundtrip_ff_buckets(bucket_vocab_embeddings_fifu):
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, "write_embeddings.fifu")
    bucket_vocab_embeddings_fifu.write(filename)
    e2 = ffp.load_finalfusion(filename)
    assert bucket_vocab_embeddings_fifu.vocab == e2.vocab
    assert bucket_vocab_embeddings_fifu.metadata == e2.metadata
    assert np.allclose(bucket_vocab_embeddings_fifu.storage, e2.storage)
    assert np.allclose(bucket_vocab_embeddings_fifu.norms, e2.norms)


def test_embeddings_pq_mmap(pq_check, embeddings_pq_memmap):
    embedding_pq = embeddings_pq_memmap.embedding("Berlin")
    embedding_fifu = pq_check.embedding("Berlin")
    assert np.allclose(embedding_fifu, embedding_pq, atol=0.3)


def test_embeddings_pq_read(pq_check, embeddings_pq_read):
    embedding_pq = embeddings_pq_read.embedding("Berlin")
    embedding_fifu = pq_check.embedding("Berlin")
    assert np.allclose(embedding_fifu, embedding_pq, atol=0.3)


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


def test_unknown_embeddings(embeddings_fifu, bucket_vocab_embeddings_fifu):
    assert embeddings_fifu.embedding(
        "OOV") is None, "Unknown lookup with no default failed"
    assert embeddings_fifu.embedding(
        "OOV",
        default=None) is None, "Unknown lookup with 'None' default failed"
    assert np.allclose(
        embeddings_fifu.embedding("OOV",
                                  default=np.zeros(10, dtype=np.float32)),
        np.array([0.] * 10)), "Unknown lookup with 'list' default failed"
    out = np.zeros(10, dtype=np.float32)
    default = np.ones(10, dtype=np.float32)
    out2 = embeddings_fifu.embedding("OOV", default=default, out=out)
    assert out is out2
    assert np.allclose(out, default)
    out2 = embeddings_fifu.embedding("OOV", default=0, out=out)
    assert np.allclose(out2, 0)
    with pytest.raises(TypeError):
        _ = bucket_vocab_embeddings_fifu.embedding(None)


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
    embeddings = ffp.Embeddings(vocab=ffp.vocab.SimpleVocab(vocab),
                                storage=ffp.storage.NdArray(matrix))
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


def test_read_fasttext(embeddings_ft):
    target_sims = np.array([
        0.6032760739326477, 0.5803255438804626, 0.5076280236244202,
        0.4979204535484314, 0.4824579954147339, 0.4742707908153534,
        0.4680115878582001, 0.466314435005188, 0.46247801184654236,
        0.4616358280181885
    ])
    target_words = [
        'man', 'oder', 'ist', 'sein', 'eines', 'online', 'wird', 'ohne',
        'kann', 'anderen'
    ]
    sims = embeddings_ft.storage[:len(embeddings_ft.vocab
                                      )] @ embeddings_ft.embedding("test")
    knn_inds = np.argpartition(sims, -10)[-10:]
    knn_inds_sorted = (-sims[knn_inds]).argsort()
    assert np.allclose(sims[knn_inds[knn_inds_sorted]], target_sims)
    assert target_words == [
        embeddings_ft.vocab.words[idx] for idx in knn_inds[knn_inds_sorted]
        if embeddings_ft.vocab.words[idx] != "test"
    ]


def test_buckets_to_explicit(bucket_vocab_embeddings_fifu):
    explicit = bucket_vocab_embeddings_fifu.bucket_to_explicit()
    assert bucket_vocab_embeddings_fifu.vocab.words == explicit.vocab.words
    for e1, e2 in zip(bucket_vocab_embeddings_fifu, explicit):
        assert e1[0] == e1[0]
        assert np.allclose(e1[1], e2[1])
    assert bucket_vocab_embeddings_fifu.vocab.idx_bound == 1024 + len(
        bucket_vocab_embeddings_fifu.vocab)
    assert explicit.vocab.idx_bound == len(
        bucket_vocab_embeddings_fifu.vocab) + 16
    bucket_indexer = bucket_vocab_embeddings_fifu.vocab.subword_indexer
    explicit_indexer = explicit.vocab.subword_indexer
    for ngram in explicit_indexer:
        assert np.allclose(
            bucket_vocab_embeddings_fifu.storage[2 + bucket_indexer(ngram)],
            explicit.storage[2 + explicit_indexer(ngram)])
