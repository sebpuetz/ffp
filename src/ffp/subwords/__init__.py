"""
Interface to subwords.

Provides access to the subword indexers and a method to extract ngrams from a given string.
"""

from .explicit_indexer import ExplicitIndexer
from .hash_indexers import FinalfusionHashIndexer, FastTextIndexer
from .ngrams import word_ngrams
