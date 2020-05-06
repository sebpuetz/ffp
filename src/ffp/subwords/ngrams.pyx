# cython: language_level=3
# cython: embedsignature=True

from libc.stdint cimport uint32_t

cpdef word_ngrams(str word, uint32_t min_n=3, uint32_t max_n=6, bint bracket=True):
    """
    Get the ngrams for the given word.
    :param word: word
    :param min_n: lower bound of n-gram range
    :param max_n: upper bound of n-gram range
    :param bracket: whether to bracket the word
    :return: list of n-grams
    """
    if word is None:
        raise TypeError("Can't extract ngrams for None type")
    assert max_n >= min_n > 0
    if bracket:
        word = "<%s>" % word
    cdef size_t i = 0
    cdef uint32_t j
    cdef size_t length = len(word)
    ngrams = []
    if length < min_n:
        return ngrams
    if length < max_n:
        max_n = length
    for i in range(length + 1 - min_n):
        for j in range(max_n, min_n-1, -1):
            if j + i <= length:
                ngrams.append(word[i:i + j])
    return ngrams
