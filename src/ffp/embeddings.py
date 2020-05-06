"""
Finalfusion Embeddings
"""
import struct
from enum import Enum, unique
from typing import Optional, Union, Tuple

import numpy as np

from ffp.io import Chunk, ChunkIdentifier, Header
import ffp.metadata
import ffp.norms
import ffp.storage
import ffp.vocab


@unique
class Format(Enum):
    """
    Supported embedding formats.
    """
    finalfusion = "finalfusion"
    finalfusion_mmap = "finalfusion_mmap"
    text = "text"
    textdims = "textdims"
    word2vec = "word2vec"


class Embeddings:  # pylint: disable=too-many-instance-attributes
    """
    Embeddings class.

    Typically consists of a storage and vocab. Other possible chunks are norms
    corresponding to the embeddings of the in-vocab tokens and metadata.

    No chunk is required, but at least one chunk needs to be present to
    construct embeddings.
    """
    def __init__(self,
                 storage: Optional[ffp.storage.Storage] = None,
                 vocab: Optional[ffp.vocab.Vocab] = None,
                 norms: Optional[ffp.norms.Norms] = None,
                 metadata: Optional[ffp.metadata.Metadata] = None):
        assert storage is None or isinstance(
            storage, ffp.storage.Storage), "storage is required to be Storage"
        assert vocab is None or isinstance(
            vocab, ffp.vocab.Vocab), "vocab is required to be Vocab"
        assert norms is None or isinstance(
            norms, ffp.norms.Norms), "norms is required to be Norms"
        assert metadata is None or isinstance(
            metadata,
            ffp.metadata.Metadata), "metadata is required to be Metadata"
        if vocab is not None and storage is not None:
            assert storage.shape[
                0] == vocab.idx_bound, "Number of embeddings needs to be equal to vocab's idx_bound"
        if vocab is not None and norms is not None:
            assert len(vocab) == len(
                norms), "Vocab length needs to be equal to number of norms."
        if storage is not None and norms is not None:
            assert storage.shape[0] >= len(
                norms
            ), "Number of embeddings needs to be greater than or equal to number of norms."
        self._storage = storage
        self._vocab = vocab
        self._norms = norms
        self._metadata = metadata

    def chunks(self):
        """
        Get the present chunks as a list.
        :return: Chunks
        """
        chunks = []
        if self._vocab is not None:
            chunks.append(self.vocab)
        if self._storage is not None:
            chunks.append(self.storage)
        if self._metadata is not None:
            chunks.append(self.metadata)
        if self._norms is not None:
            chunks.append(self.norms)
        return chunks

    @property
    def storage(self) -> Optional[ffp.storage.Storage]:
        """
        Get the storage. Returns None if no storage is set.
        :return: Storage
        """
        return self._storage

    @storage.setter
    def storage(self, storage: Optional[ffp.storage.Storage]):
        if storage is None:
            self._storage = None
        elif isinstance(storage, ffp.storage.Storage):
            if self._norms is not None:
                assert storage.shape[0] >= len(
                    self._norms
                ), "Number of embeddings needs to be greater than or equal to number of norms."
            if self._vocab is not None:
                assert storage.shape[
                    0] == self._vocab.idx_bound,\
                    "Number of embeddings needs to be equal to vocab's idx_bound"
            self._storage = storage
        else:
            raise TypeError("Expected 'None' or 'Vocab'.")

    @property
    def vocab(self) -> Optional[ffp.vocab.Vocab]:
        """
        Get the vocab. Returns None if no vocab is set.
        :return: Vocab
        """
        return self._vocab

    @vocab.setter
    def vocab(self, vocab: Optional[ffp.vocab.Vocab]):
        if vocab is None:
            self._vocab = None
        elif isinstance(vocab, ffp.vocab.Vocab):
            if self._norms is not None:
                assert len(vocab) == len(
                    self._norms
                ), "Vocab length needs to be equal to number of norms."
            if self._storage is not None:
                # pylint: disable=unsubscriptable-object
                assert self._storage.shape[
                    0] == vocab.idx_bound, \
                    "Vocab's idx_bound needs to be equal to number of embeddings."
            self._vocab = vocab
        else:
            raise TypeError("Expected 'None' or 'Vocab'.")

    @property
    def norms(self) -> Optional[ffp.norms.Norms]:
        """
        Get the norms. Returns None if no norms are set.
        :return: Norms
        """
        return self._norms

    @norms.setter
    def norms(self, norms: Optional[ffp.norms.Norms]):
        if norms is None:
            self._norms = None
        elif isinstance(norms, ffp.norms.Norms):
            if self._vocab is not None:
                assert len(self._vocab) == len(
                    norms), "Vocab and norms need to have same length"
            if self._storage is not None:
                # pylint: disable=unsubscriptable-object
                assert self._storage.shape[0] >= len(
                    norms
                ), "Number of norms needs to be equal to or less than number of embeddings"
            self._norms = norms
        else:
            raise TypeError("Expected 'None' or 'Norms'.")

    @property
    def metadata(self) -> Optional[ffp.metadata.Metadata]:
        """
        Get the metadata. Returns None if no metadata is set.
        :return: Metadata
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[ffp.metadata.Metadata]):
        if metadata is None:
            self._metadata = None
        elif isinstance(metadata, ffp.metadata.Metadata):
            self._metadata = metadata
        else:
            raise TypeError("Expected 'None' or 'Metadata'.")

    def write(self, path: str):
        """
        Write the Embeddings in finalfusion format to the given path.
        :param path: path
        """
        with open(path, 'wb') as file:
            chunks = self.chunks()
            header = Header([chunk.chunk_identifier() for chunk in chunks])
            header.write_chunk(file)
            for chunk in chunks:
                chunk.write_chunk(file)

    def embedding(self,
                  word: str,
                  out: Optional[np.ndarray] = None,
                  default: Optional[np.ndarray] = None
                  ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Look up the embedding for the given word.
        :param word: query token
        :param out: Optional array that the embedding should be written to.
        :param default: default value to return if no embedding was found.
        :return: the embedding
        """
        idx = self._vocab.idx(word)
        if idx is None:
            return default
        res = self._storage[idx]
        if res.ndim == 1:
            if out is not None:
                out[:] = res
            else:
                out = res
        else:
            out = np.add.reduce(res, 0, out=out, keepdims=False)
            out /= np.linalg.norm(out)

        return out

    def embedding_with_norm(self,
                            word: str,
                            out: Optional[np.ndarray] = None,
                            default: Optional[Tuple[np.ndarray, float]] = None
                            ) -> Tuple[np.ndarray, float]:
        """
        Look up the embedding for the given word together with its norm.
        :param word: query token
        :param out: Optional array that the embedding should be written to.
        :param default: default value to return if no embedding was found.
        :return: tuple containing the embedding and the norm
        """
        if self._norms is None:
            raise TypeError("embeddings don't contain norms chunk")
        idx = self._vocab.idx(word)
        if idx is None:
            return default
        res = self._storage[idx]
        if res.ndim == 1:
            if out is not None:
                out[:] = res
            else:
                out = res
            return out, self._norms[idx]

        out = np.add.reduce(res, 0, out=out, keepdims=False)
        norm = np.linalg.norm(out)
        out /= norm
        return out, norm

    def bucket_to_explicit(self) -> 'Embeddings':
        """
        Convert bucket embeddings to embeddings with explicit lookup.

        Multiple embeddings can still map to the same bucket, but all buckets that are not
        indexed by in-vocabulary ngrams are eliminated. This can have a big impact on the size of
        the embedding matrix.

        :return: Embeddings with an explicit ngram lookup.
        """
        bucket_vocabs = (ffp.vocab.FastTextVocab,
                         ffp.vocab.FinalfusionBucketVocab)
        if not isinstance(self._vocab, bucket_vocabs):
            raise TypeError(
                "Only bucketed embeddings can be converted to explicit.")
        vocab = self._vocab.to_explicit()
        if self._storage is None:
            return Embeddings(vocab=vocab)
        storage = np.zeros((vocab.idx_bound, self._storage.shape[1]),
                           dtype=np.float32)
        storage[:len(vocab)] = self._storage[:len(vocab)]
        for ngram in vocab.subword_indexer:
            storage[len(vocab) + vocab.subword_indexer(ngram)] = self._storage[
                len(vocab) + self._vocab.subword_indexer(ngram)]
        return Embeddings(vocab=vocab, storage=ffp.storage.NdArray(storage))

    def __getitem__(self, item):
        # no need to check for none since Vocab raises KeyError if it can't produce indices
        idx = self._vocab[item]
        res = self._storage[idx]
        if res.ndim == 1:
            return res
        embed_sum = res.sum(axis=0)
        return embed_sum / np.linalg.norm(embed_sum)

    def __contains__(self, item):
        if self._vocab is None:
            raise TypeError("These embeddings don't contain a vocabulary")
        return item in self._vocab

    def __repr__(self):
        return "Embeddings { \n" + "\n".join(
            [repr(chunk) for chunk in self.chunks()]) + "\n}"

    def __iter__(self):
        if self._norms is not None:
            return zip(self._vocab.words, self._storage, self._norms)
        return zip(self._vocab.words, self._storage)


def load_finalfusion(path: str, mmap: bool = False) -> Embeddings:
    """
    Read embeddings from a file in finalfusion format.
    :param path: path
    :param mmap: whether to mmap the storage
    :return: Embeddings
    """
    with open(path, 'rb') as file:
        _ = Header.read_chunk(file)
        chunk_id, _ = Chunk.read_chunk_header(file)
        embeddings = Embeddings()
        while True:
            if chunk_id.is_storage():
                embeddings.storage = _STORAGE_READERS[chunk_id](file, mmap)
            elif chunk_id.is_vocab():
                embeddings.vocab = _VOCAB_READERS[chunk_id](file)
            elif chunk_id == ChunkIdentifier.NdNorms:
                embeddings.norms = ffp.norms.Norms.read_chunk(file)
            elif chunk_id == ChunkIdentifier.Metadata:
                embeddings.metadata = ffp.metadata.Metadata.read_chunk(file)
            else:
                chunk_id, _ = Chunk.read_chunk_header(file)
                raise TypeError("Unknown chunk type: " + str(chunk_id))
            chunk_header = Chunk.read_chunk_header(file)
            if chunk_header is None:
                break
            chunk_id, _ = chunk_header
        return embeddings


def load_word2vec(path: str) -> Embeddings:
    """
    Read embeddings in word2vec binary format:
    Files are expected to start with a line containing rows and cols in utf-8. Words are encoded
    in utf-8 followed by a single whitespace. After the whitespace the embedding components are
    expected as little-endian float32.
    :param path: path
    :return: Embeddings
    """
    words = []
    with open(path, 'rb') as file:
        rows, cols = map(int, file.readline().decode("utf-8").split())
        matrix = np.zeros((rows, cols), dtype=np.float32)
        for row in range(rows):
            word = []
            while True:
                byte = file.read(1)
                if byte == b' ':
                    break
                if byte == b'':
                    raise EOFError
                if byte != b'\n':
                    word.append(byte)
            word = b''.join(word).decode('utf-8')
            words.append(word)
            vec = file.read(cols * matrix.itemsize)
            matrix[row] = np.frombuffer(vec, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    matrix /= np.expand_dims(norms, axis=1)
    return Embeddings(storage=ffp.storage.NdArray(matrix),
                      norms=ffp.norms.Norms(norms),
                      vocab=ffp.vocab.SimpleVocab(words))


def load_textdims(path) -> Embeddings:
    """
    Read emebddings in textdims format:
    The first line contains whitespace separated rows and cols, the rest of the file contains
    whitespace separated word and vector components.
    :param path:
    :return:
    """
    words = []
    with open(path) as file:
        rows, cols = next(file).split()
        matrix = np.zeros((int(rows), int(cols)), dtype=np.float32)
        for i, line in enumerate(file):
            line = line.strip().split()
            words.append(line[0])
            matrix[i] = line[1:]
    norms = np.linalg.norm(matrix, axis=1)
    matrix /= np.expand_dims(norms, axis=1)
    return Embeddings(storage=ffp.storage.NdArray(matrix),
                      norms=ffp.norms.Norms(norms),
                      vocab=ffp.vocab.SimpleVocab(words))


def load_text(path: str) -> Embeddings:
    """
    Read embeddings in text format:
    Each line contains a word followed by a whitespace and a list of whitespace separated
    values.
    :param path: path
    :return: Embeddings
    """
    words = []
    vecs = []
    with open(path) as file:
        for line in file:
            line = line.strip().split()
            words.append(line[0])
            vecs.append(line[1:])
        matrix = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    matrix /= np.expand_dims(norms, axis=1)
    return Embeddings(storage=ffp.storage.NdArray(matrix),
                      norms=ffp.norms.Norms(norms),
                      vocab=ffp.vocab.SimpleVocab(words))


def load_fastText(path: str) -> Embeddings:  # pylint: disable=invalid-name
    """
    Read embeddings from a file in fastText format.
    :param path: path
    :return: Embeddings
    """
    with open(path, 'rb') as file:
        _read_ft_header(file)
        metadata = _read_ft_cfg(file)
        vocab = _read_ft_vocab(file, metadata['buckets'], metadata['min_n'],
                               metadata['max_n'])
        quantized = struct.unpack("<B", file.read(1))[0]
        if quantized:
            raise NotImplementedError("Quantized storage is not supported")
        rows, cols = struct.unpack("<QQ", file.read(16))
        matrix = np.fromfile(file=file, count=rows * cols, dtype=np.float32)
        matrix = np.reshape(matrix, (rows, cols))
        for i, word in enumerate(vocab):
            indices = [i] + vocab.subword_indices(word)
            matrix[i] = matrix[indices].mean(0, keepdims=False)
        norms = np.linalg.norm(matrix[:len(vocab)], axis=1)
        matrix[:len(vocab)] /= np.expand_dims(norms, axis=1)
        storage = ffp.storage.NdArray(matrix)
        norms = ffp.norms.Norms(norms)
    return Embeddings(storage, vocab, norms, metadata)


def _read_ft_header(file):
    magic = struct.unpack("<I", file.read(4))[0]
    if magic != 793_712_314:
        raise IOError("Magic should be 793_712_314, not: " + str(magic))
    version = struct.unpack("<I", file.read(4))[0]
    if version > 12:
        raise ValueError("Expected version 12, got: " + str(version))


def _read_ft_cfg(file):
    cfg = struct.unpack("<" + 12 * "I" + "d", file.read(12 * 4 + 8))
    loss, model = cfg[6:8]  # map to string
    if loss == 1:
        loss = 'HierarchicalSoftmax'
    elif loss == 2:
        loss = 'NegativeSampling'
    elif loss == 3:
        loss = 'Softmax'
    if model == 1:
        model = 'CBOW'
    elif model == 2:
        model = 'SkipGram'
    elif model == 3:
        model = 'Supervised'
    metadata = ffp.metadata.Metadata({
        'dims': cfg[0],
        'window_size': cfg[1],
        'epoch': cfg[2],
        'min_count': cfg[3],
        'ns': cfg[4],
        'word_ngrams': cfg[5],
        'loss': loss,
        'model': model,
        'buckets': cfg[8],
        'min_n': cfg[9],
        'max_n': cfg[10],
        'lr_update_rate': cfg[11],
        'sampling_threshold': cfg[12],
    })
    return metadata


def _read_ft_vocab(file, buckets, min_n, max_n):
    vocab_size = struct.unpack("<I", file.read(4))[0]
    _ = struct.unpack("<I", file.read(4))[0]  # discard n_words
    n_labels = struct.unpack("<I", file.read(4))[0]
    if n_labels:
        raise NotImplementedError(
            "fastText prediction models are not supported")
    _n_tokens = struct.unpack("<Q", file.read(8))[0]
    prune_idx_size = struct.unpack("<q", file.read(8))[0]
    if prune_idx_size > 0:
        raise NotImplementedError("Pruned vocabs are not supported")
    words = []
    for _ in range(vocab_size):
        word = bytearray()
        while True:
            byte = file.read(1)
            if byte == b'\x00':
                words.append(word.decode("utf8"))
                break
            if byte == b'':
                raise EOFError
            word.extend(byte)
        _freq = struct.unpack("<Q", file.read(8))[0]
        entry_type = struct.unpack("<B", file.read(1))[0]
        if entry_type != 0:
            raise ValueError("Non word entry", word)
    indexer = ffp.subwords.FastTextIndexer(buckets, min_n, max_n)
    return ffp.vocab.FastTextVocab(words, indexer)


_VOCAB_READERS = {
    ChunkIdentifier.SimpleVocab: ffp.vocab.SimpleVocab.read_chunk,
    ChunkIdentifier.BucketSubwordVocab:
    ffp.vocab.FinalfusionBucketVocab.read_chunk,
    ChunkIdentifier.FastTextSubwordVocab: ffp.vocab.FastTextVocab.read_chunk,
    ChunkIdentifier.ExplicitSubwordVocab: ffp.vocab.ExplicitVocab.read_chunk,
}

_STORAGE_READERS = {
    ChunkIdentifier.NdArray: ffp.storage.NdArray.load,
    ChunkIdentifier.QuantizedArray: ffp.storage.QuantizedArray.load,
}
