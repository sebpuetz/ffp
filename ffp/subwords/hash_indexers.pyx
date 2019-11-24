# cython: language_level=3
# cython: embedsignature=True

from libc.stdint cimport int8_t, uint8_t, uint32_t, uint64_t, UINT32_MAX

cdef class FinalfusionHashIndexer:
    """
    FinalfusionHashIndexer
    """
    cdef public uint32_t min_n
    cdef public uint32_t max_n
    cdef public uint64_t buckets_exp
    cdef uint64_t mask

    def __init__(self, bucket_exp=21, min_n=3, max_n=6):
        assert max_n >= min_n > 0, "min_n needs to be greater than 0 and less than max_n"
        assert 0 < bucket_exp <= 64, "bucket_exp needs to be greater than 0 and less than 65"
        self.min_n = min_n
        self.max_n = max_n
        self.buckets_exp = bucket_exp
        self.mask = ((1 << bucket_exp) - 1)

    def __call__(self, str ngram):
        return fifu_hash_ngram(ngram, len(ngram)) & self.mask

    @property
    def idx_bound(self) -> int:
        return pow(2, self.buckets_exp)

    cpdef subword_indices(self, str word, uint64_t offset = 0, bint bracket=True, bint with_ngrams=False):
        """
        Get the subword indices for the given word.

        :param word: the word
        :param offset: is added to each index
        :param bracket: whether to bracket the word with '<' and '>'
        :param with_ngrams: whether to return the indices with corresponding ngrams
        :return: List of subword indices, obtionally as tuples with ngrams
        """
        if bracket:
            word = "<%s>" % word
        cdef uint32_t j
        cdef size_t length = len(word)
        cdef size_t i = 0
        cdef str ngram
        cdef uint64_t h
        cdef uint32_t max_n = self.max_n
        ngrams = []
        if length < self.min_n:
            return ngrams
        if length < max_n:
            max_n = length
        for i in range(length + 1 - self.min_n):
            for j in range(max_n, self.min_n-1, -1):
                if j + i <= length:
                    ngram = word[i:i + j]
                    h = (fifu_hash_ngram(ngram, j) & self.mask) + offset
                    if with_ngrams:
                        ngrams.append((ngram, h))
                    else:
                        ngrams.append(h)
        return ngrams

cdef class FastTextIndexer:
    """
    FastTextIndexer
    """
    cdef public uint32_t min_n
    cdef public uint32_t max_n
    cdef public uint64_t n_buckets

    def __init__(self, n_buckets=2000000, min_n=3, max_n=6):
        assert max_n >= min_n > 0, "min_n needs to be greater than 0 and less than max_n"
        assert 0 < n_buckets <= UINT32_MAX, "number of buckets needs be between 0 and pow(2, 32)"
        self.min_n = min_n
        self.max_n = max_n
        self.n_buckets = n_buckets

    def __call__(self, str ngram):
        return ft_hash_ngram(ngram) % self.n_buckets

    cpdef subword_indices(self,
                          str word,
                          uint64_t offset=0,
                          bint bracket=True,
                          bint with_ngrams=False):
        """
        Get the subword indices for the given word.

        :param word: the word
        :param offset: is added to each index
        :param bracket: whether to bracket the word with '<' and '>'
        :param with_ngrams: whether to return the indices with corresponding ngrams
        :return: List of subword indices, obtionally as tuples with ngrams
        """
        if bracket:
            word = "<%s>" % word
        cdef uint32_t j
        cdef size_t length = len(word)
        cdef size_t i = 0
        cdef str ngram
        cdef uint64_t h
        cdef uint32_t max_n = self.max_n
        ngrams = []
        if length < self.min_n:
            return ngrams
        if length < self.max_n:
            max_n = length
        for i in range(length + 1 - self.min_n):
            for j in range(max_n, self.min_n-1, -1):
                if j + i > length:
                    continue
                ngram = word[i:i + j]
                h = (ft_hash_ngram(ngram) % self.n_buckets) + offset
                if with_ngrams:
                    ngrams.append((ngram, h))
                else:
                    ngrams.append(h)
        return ngrams

    @property
    def idx_bound(self) -> int:
        return self.n_buckets

cdef uint64_t SEED64 = 0xcbf29ce484222325
cdef uint64_t PRIME64 = 0x100000001b3
cdef uint32_t SEED32 = 2166136261
cdef uint32_t PRIME32 = 16777619

cdef uint64_t fifu_hash_ngram(str ngram, const size_t n_chars):
    cdef uint32_t i = 0
    cdef uint64_t h = SEED64
    cdef uint64_t c

    h = fnv64(<uint8_t*> &n_chars, 8, h=h)
    for i in range(n_chars):
        c = ord(ngram[i])
        h = fnv64(<uint8_t*> &c, 4, h=h)
        i += 1
    return h

cdef uint64_t fnv64(const uint8_t*data,
                    const uint32_t n_bytes,
                    uint64_t h):
    cdef uint32_t i = 0
    for i in range(n_bytes):
        h ^= <uint32_t> (<uint8_t> data[i])
        h *= PRIME64
        i += 1
    return h

cdef uint64_t ft_hash_ngram(str ngram):
    cdef bytes utf8_ngram = ngram.encode("utf-8")
    cdef uint32_t n_bytes = len(utf8_ngram)
    cdef const unsigned char*b_ngram = utf8_ngram
    cdef uint32_t i = 0
    cdef uint32_t h = SEED32
    while i < n_bytes:
        h ^= <uint32_t> (<int8_t> utf8_ngram[i])
        h *= PRIME32
        i += 1
    return h
