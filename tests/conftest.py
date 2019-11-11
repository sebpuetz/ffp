import os

import pytest


@pytest.fixture
def tests_root():
    yield os.path.dirname(__file__)
