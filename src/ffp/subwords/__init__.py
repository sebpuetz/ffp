"""
Interface to subwords.

Provides access to the subword indexers and a method to extract ngrams from a given string.
"""

from ffp.subwords.explicit_indexer import ExplicitIndexer
from ffp.subwords.hash_indexers import FinalfusionHashIndexer, FastTextIndexer
from ffp.subwords.ngrams import word_ngrams
