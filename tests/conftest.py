import os

import ffp
import numpy as np
import pytest


@pytest.fixture
def simple_vocab_embeddings_fifu(tests_root):
    yield ffp.load_finalfusion(os.path.join(tests_root,
                                            "data/simple_vocab.fifu"),
                               mmap=False)


@pytest.fixture
def vocab_array_tuple(tests_root):
    with open(os.path.join(tests_root, "data", "embeddings.txt")) as f:
        lines = f.readlines()
        v = []
        m = []
        for line in lines:
            line = line.split()
            v.append(line[0])
            m.append([float(p) for p in line[1:]])
    return v, np.array(m, dtype=np.float32)


@pytest.fixture
def simple_vocab_fifu(tests_root):
    yield ffp.vocab.load_vocab(
        os.path.join(tests_root, "data/simple_vocab.fifu"))


@pytest.fixture
def bucket_vocab_embeddings_fifu(tests_root):
    yield ffp.load_finalfusion(os.path.join(tests_root,
                                            "data/ff_buckets.fifu"))


@pytest.fixture
def embeddings_text(tests_root):
    yield ffp.embeddings.load_text(
        os.path.join(tests_root, "data/embeddings.txt"))


@pytest.fixture
def embeddings_text_dims(tests_root):
    yield ffp.embeddings.load_textdims(
        os.path.join(tests_root, "data/embeddings.dims.txt"))


@pytest.fixture
def embeddings_fifu(tests_root):
    yield ffp.load_finalfusion(os.path.join(tests_root,
                                            "data/embeddings.fifu"))


@pytest.fixture
def embeddings_w2v(tests_root):
    yield ffp.embeddings.load_word2vec(
        os.path.join(tests_root, "data/embeddings.w2v"))


@pytest.fixture
def embeddings_ft(tests_root):
    yield ffp.embeddings.load_fastText(
        os.path.join(tests_root, "data/fasttext.bin"))


@pytest.fixture
def tests_root():
    yield os.path.dirname(__file__)
