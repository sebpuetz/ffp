import os

import ffp
import numpy as np
import pytest


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
    yield ffp.vocab.Vocab.read(
        os.path.join(tests_root, "data/simple_vocab.fifu"))


@pytest.fixture
def tests_root():
    yield os.path.dirname(__file__)
