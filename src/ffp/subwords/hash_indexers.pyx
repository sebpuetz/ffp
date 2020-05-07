# cython: language_level=3
# cython: embedsignature=True
cimport cython
from cpython cimport array

from libc.stdint cimport int8_t, uint8_t, uint32_t, uint64_t, UINT32_MAX

# subword_indices methods could be optimized by calculating the number of ngrams and preallocating an array.array:
# cdef size_t n_ngrams = <size_t> (0.5 * (-1. + min_n - max_n)*(min_n + max_n - 2. * (1. + length)) )
# cdef array.array result = array.array('Q')
# array.resize(result, n_ngrams)
# downside of this is returning an array in place of list. Speedup ~30%

cdef class FinalfusionHashIndexer:
    """
    FinalfusionHashIndexer

    FinalfusionHashIndexer is a hash-based subword indexer. It hashes n-grams with the FNV-1a
    algorithm and maps the hash to a predetermined bucket space.

    N-grams can be indexed directly through the `__call__` method or all n-grams in a string
    can be indexed in bulk through the `subword_indices` method.
    """
    cdef public uint32_t min_n
    cdef public uint32_t max_n
    cdef public uint64_t buckets_exp
    cdef uint64_t mask

    def __init__(self, bucket_exp=21, min_n=3, max_n=6):
        assert max_n >= min_n > 0, \
            f"Min_n ({min_n}) must be greater than 0, max_n ({min_n}) must be >= min_n"
        assert 0 < bucket_exp <= 64, \
            f"bucket_exp ({bucket_exp}) needs to be greater than 0 and less than 65"
        self.min_n = min_n
        self.max_n = max_n
        self.buckets_exp = bucket_exp
        self.mask = ((1 << bucket_exp) - 1)

    def __call__(self, str ngram):
        return fifu_hash_ngram(ngram, 0, len(ngram)) & self.mask

    @property
    def idx_bound(self) -> int:
        """
        Get the **exclusive** upper bound

        This is the number of distinct indices.

        Returns
        -------
        idx_bound : int
            Exclusive upper bound of the indexer.
        """
        return pow(2, self.buckets_exp)

    cpdef subword_indices(self, str word, uint64_t offset = 0, bint bracket=True, bint with_ngrams=False):
        """
        Get the subword indices for a word.
        
        Parameters
        ----------
        word : str
            The string to extract n-grams from
        offset : int
            The offset to add to the index, e.g. the length of the word-vocabulary.
        bracket : bool
            Toggles bracketing the input string with `<` and `>`
        with_ngrams : bool
            Toggles returning tuples of (ngram, idx)
        
        Returns
        -------
        indices : list
            List of n-gram indices, optionally as `(str, int)` tuples.
        
        Raises
        ------
        TypeError
            If `word` is None.
        """
        if word is None:
            raise TypeError("Can't extract ngrams for None type")
        if bracket:
            word = "<%s>" % word
        cdef uint32_t j
        cdef Py_ssize_t length = len(word)
        cdef Py_ssize_t i
        cdef uint64_t h
        cdef list ngrams = []
        if length < self.min_n:
            return ngrams
        cdef uint32_t max_n = min(length, self.max_n)
        # iterate over starting points
        for i in range(length + 1 - self.min_n):
            # iterate over ngram lengths, long to short
            for j in range(max_n, self.min_n - 1, -1):
                if j + i <= length:
                    h = (fifu_hash_ngram(word, i, j) & self.mask) + offset
                    if with_ngrams:
                        ngrams.append((h, word[i:i + j]))
                    else:
                        ngrams.append(h)
        return ngrams

cdef class FastTextIndexer:
    """
    FastTextIndexer

    FastTextIndexer is a hash-based subword indexer. It hashes n-grams with (a slightly) FNV-1a
    variant and maps the hash to a predetermined bucket space.

    N-grams can be indexed directly through the `__call__` method or all n-grams in a string
    can be indexed in bulk through the `subword_indices` method.
    """
    cdef public uint32_t min_n
    cdef public uint32_t max_n
    cdef public uint64_t n_buckets

    def __init__(self, n_buckets=2000000, min_n=3, max_n=6):
        assert max_n >= min_n > 0, \
            f"Min_n ({min_n}) must be greater than 0, max_n ({min_n}) must be >= min_n"
        assert 0 < n_buckets <= UINT32_MAX, \
            f"n_buckets ({n_buckets}) needs to be between 0 and {pow(2, 32)}"
        self.min_n = min_n
        self.max_n = max_n
        self.n_buckets = n_buckets

    def __call__(self, str ngram):
        cdef bytes b_ngram = ngram.encode("utf8")
        return ft_hash_ngram(b_ngram, 0, len(b_ngram)) % self.n_buckets

    @cython.cdivision(True)
    cpdef subword_indices(self,
                          str word,
                          uint64_t offset=0,
                          bint bracket=True,
                          bint with_ngrams=False):
        """
        Get the subword indices for a word.
        
        Parameters
        ----------
        word : str
            The string to extract n-grams from
        offset : int
            The offset to add to the index, e.g. the length of the word-vocabulary.
        bracket : bool
            Toggles bracketing the input string with `<` and `>`
        with_ngrams : bool
            Toggles returning tuples of (ngram, idx)
        
        Returns
        -------
        indices : list
            List of n-gram indices, optionally as `(str, int)` tuples.
        
        Raises
        ------
        TypeError
            If `word` is None.
        """
        if word is None:
            raise TypeError("Can't extract ngrams for None type")
        cdef unsigned int start, end
        cdef Py_ssize_t i, j
        cdef uint64_t h
        if bracket:
            word = "<%s>" % word
        cdef bytes b_word = word.encode("utf-8")
        cdef const unsigned char* b_word_view = b_word
        cdef Py_ssize_t length = len(word)
        cdef uint32_t max_n = min(self.max_n, length)
        cdef array.array offsets = find_utf8_boundaries(b_word_view, len(b_word))
        cdef list ngrams = []
        # iterate over starting points by character
        for i in range(length + 1 - self.min_n):
            # offsets[i] corresponds to the byte offset to the start of the char at word[i]
            start = offsets.data.as_uints[i]
            # iterate over ngram lengths, long to short
            for j in range(max_n, self.min_n - 1, -1):
                if j + i <= length:
                    # offsets[i+j] holds the exclusive upper bounds of the bytes for the character word[i+j-1]
                    end = offsets.data.as_uints[i + j]
                    h = ft_hash_ngram(b_word_view, start, end) % self.n_buckets + offset
                    if with_ngrams:
                        ngrams.append((h, word[i:i + j]))
                    else:
                        ngrams.append(h)
        return ngrams

    @property
    def idx_bound(self) -> int:
        return self.n_buckets

@cython.boundscheck(False)
cdef array.array find_utf8_boundaries(const unsigned char* w, const Py_ssize_t n_bytes):
    cdef Py_ssize_t b
    cdef Py_ssize_t i = 0
    cdef array.array offsets = array.array('I')
    # n_bytes + 1 to store n_bytes as final boundary
    array.resize(offsets, n_bytes + 1)
    for b in range(n_bytes):
        # byte w[b] is not a continuation byte, therefore beginning of char; store offset
        if (w[b] & 0xC0) != 0x80:
            offsets.data.as_uints[i] = b
            i += 1
    offsets.data.as_uints[i] = <unsigned int> n_bytes
    return offsets

cdef uint64_t SEED64 = 0xcbf29ce484222325
cdef uint64_t PRIME64 = 0x100000001b3
cdef uint32_t SEED32 = 2166136261
cdef uint32_t PRIME32 = 16777619

@cython.boundscheck(False)
cdef uint64_t fifu_hash_ngram(str word, const Py_ssize_t start, const Py_ssize_t length):
    cdef uint32_t c
    cdef Py_ssize_t i = 0
    cdef uint64_t h = SEED64

    h = fnv64(<uint8_t*> &length, 8, h=h)
    for i in range(start, start + length):
        # extract unicode for char, cast to u8 pointer for hashing extracting bytes manually from memoryview/bytes is
        # more complex because of handling prefixes.
        c = ord(word[i])
        h = fnv64(<uint8_t*> &c, 4, h=h)
    return h

cdef uint64_t fnv64(const uint8_t* data,
                    const Py_ssize_t n_bytes,
                    uint64_t h):
    cdef Py_ssize_t i = 0
    for i in range(n_bytes):
        h ^= <uint32_t> (<uint8_t> data[i])
        h *= PRIME64
        i += 1
    return h

cdef uint64_t ft_hash_ngram(const unsigned char* b_word, const Py_ssize_t start, const Py_ssize_t end):
    cdef Py_ssize_t i
    cdef uint32_t h = SEED32
    # iterate over bytes in range start..end and hash each byte
    for i in range(start, end):
        h ^= <uint32_t> (<int8_t> b_word[i])
        h *= PRIME32
    return h
