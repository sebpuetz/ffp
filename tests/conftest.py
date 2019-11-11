import os

import ffp
import pytest


@pytest.fixture
def simple_vocab_fifu(tests_root):
    yield ffp.vocab.Vocab.read(
        os.path.join(tests_root, "data/simple_vocab.fifu"))


@pytest.fixture
def tests_root():
    yield os.path.dirname(__file__)
