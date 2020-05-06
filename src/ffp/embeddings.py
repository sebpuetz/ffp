"""
Finalfusion Embeddings
"""
import struct
from typing import Optional, Union, Tuple

import numpy as np

from ffp.io import Chunk, ChunkIdentifier, Header
from ffp.metadata import Metadata
from ffp.norms import Norms
from ffp.storage import Storage, NdArray, QuantizedArray
from ffp.subwords import FastTextIndexer
from ffp.vocab import Vocab, FastTextVocab, FinalfusionBucketVocab, SimpleVocab, ExplicitVocab


class Embeddings:  # pylint: disable=too-many-instance-attributes
    """
    Embeddings class.

    Typically consists of a storage and vocab. Other possible chunks are norms
    corresponding to the embeddings of the in-vocab tokens and metadata.
    """
    def __init__(self,
                 storage: Optional[Storage] = None,
                 vocab: Optional[Vocab] = None,
                 norms: Optional[Norms] = None,
                 metadata: Optional[Metadata] = None):
        """
        Initialize Embeddings.

        Initializes Embeddings with the given chunks.

        Parameters
        ----------
        storage : Optional[Storage]
            Embeddings Storage.
        vocab : Optional[Vocab]
            Embeddings Vocabulary.
        norms : Optional[Norms]
            Embeddings Norms.
        metadata : Optional[Metadata]
            Embeddings Metadata.

        Raises
        ------
        AssertionError
            * if any of the chunks is not the expected chunk
            * vocab and storage are passed, but vocab.idx_bound doesn't match storage.shape[0]
            * vocab and norms are passed, but len(vocab) and len(norms) don't match
            * norms and storage are passed, but storage.shape[0] is smaller than len(norms)
        """
        assert storage is None or isinstance(
            storage, Storage), "storage is required to be Storage"
        assert vocab is None or isinstance(
            vocab, Vocab), "vocab is required to be Vocab"
        assert norms is None or isinstance(
            norms, Norms), "norms is required to be Norms"
        assert metadata is None or isinstance(
            metadata, Metadata), "metadata is required to be Metadata"
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
        Get the Embeddings Chunks as a list.

        The Chunks are ordered in the expected serialization order:
        1. Metadata
        2. Vocabulary
        3. Storage
        4. Norms

        Returns
        -------
        chunks : List[Chunk]
            List of embeddings chunks.
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
    def storage(self) -> Optional[Storage]:
        """
        Get the storage.

        Returns None if no storage is set.

        Returns
        -------
        storage : Optional[Storage]
            The embeddings storage.
        """
        return self._storage

    @storage.setter
    def storage(self, storage: Optional[Storage]):
        """
        Set the Storage.

        Parameters
        ----------
        storage : Optional[Storage]
            The new embeddings storage or None.

        Raises
        ------
        AssertionError
            * if a vocab is present and `storage.shape[0]` does not match `vocab.idx_bound`.
            * if norms are present and `storage.shape[0]` is smaller than `len(norms)`.
        TypeError
            If storage is neither a Storage nor None.
        """
        if storage is None:
            self._storage = None
        elif isinstance(storage, Storage):
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
    def vocab(self) -> Optional[Vocab]:
        """
        Get the Vocab.

        Returns None if no vocab is set.

        Returns
        -------
        vocab : Optional[Vocab]
            The embeddings vocabulary.
        """
        return self._vocab

    @vocab.setter
    def vocab(self, vocab: Optional[Vocab]):
        """
        Set the Vocab.

        Parameters
        ----------
        vocab : Optional[Vocab]
            The new embeddings vocabulary or None.

        Raises
        ------
        AssertionError
            * if a storage is present and `storage.shape[0]` does not match `vocab.idx_bound`.
            * if a norms are present and `len(norms)` does not match `len(vocab)`.
        TypeError
            If vocab is neither a Vocab nor None.
        """
        if vocab is None:
            self._vocab = None
        elif isinstance(vocab, Vocab):
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
    def norms(self) -> Optional[Norms]:
        """
        Get the Norms.

        Returns None if no norms are set.

        Returns
        -------
        norms : Optional[Norms]
            The embedding norms.
        """
        return self._norms

    @norms.setter
    def norms(self, norms: Optional[Norms]):
        """
        Set the Norms.

        Parameters
        ----------
        norms : Optional[Norms]
            The new embeddings Norms or None.

        Raises
        ------
        AssertionError
            * if a storage is present and `storage.shape[0]` is smaller than `len(norms)`.
            * if a vocab is present and `len(norms)` does not match `len(vocab)`.
        TypeError
            If norms is neither Norms nor None.
        """
        if norms is None:
            self._norms = None
        elif isinstance(norms, Norms):
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
    def metadata(self) -> Optional[Metadata]:
        """
        Get the Metadata.

        Returns None if no norms are set.

        Returns
        -------
        metadata : Optional[Metadata]
            The embeddings metadata.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[Metadata]):
        """
        Set the Metadata.

        Parameters
        ----------
        metadata : Optional[Metadata]
            The new embeddings metadata or None.

        Raises
        ------
        TypeError
            If metadata is neither Metadata nor None.
        """
        if metadata is None:
            self._metadata = None
        elif isinstance(metadata, Metadata):
            self._metadata = metadata
        else:
            raise TypeError("Expected 'None' or 'Metadata'.")

    def write(self, path: str):
        """
        Write the Embeddings to the given path.

        Writes the Embeddings to a finalfusion file at the given path.

        Parameters
        ----------
        path : str
            Path of the output file.
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
        Embedding lookup.

        Looks up the embedding for the input word.

        If an `out` array is specified, the embedding is written into the array.

        If it is not possible to retrieve an embedding for the input word, the `default`
        value is returned. This defaults to `None`. An embedding can not be retrieved if
        the vocabulary cannot provide an index for `word`.

        This method fails if either the storage or vocab are not set.

        Parameters
        ----------
        word : str
            The query word.
        out : Optional[numpy.ndarray]
            Optional output array to write the embedding into.
        default: Optional[numpy.ndarray]
            Optional default value to return if no embedding can be retrieved. Defaults to None.

        Returns
        -------
        embedding : Optional[numpy.ndarray]
            The retrieved embedding or the default value.
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
        Embedding lookup.

        Looks up the embedding for the input word together with its norm.

        If an `out` array is specified, the embedding is written into the array.

        If it is not possible to retrieve an embedding for the input word, the `default`
        value is returned. This defaults to `None`. An embedding can not be retrieved if
        the vocabulary cannot provide an index for `word`.

        This method fails if either storage, vocab or norms are not set.

        Parameters
        ----------
        word : str
            The query word.
        out : Optional[numpy.ndarray]
            Optional output array to write the embedding into.
        default: Optional[numpy.ndarray]
            Optional default value to return if no embedding can be retrieved. Defaults to None.

        Returns
        -------
        (embedding, norm) : Tuple[Optional[numpy.ndarray], float]
            Tuple with the retrieved embedding or the default value at the first index and the
            norm at the second index.
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
        indexed by in-vocabulary n-grams are eliminated. This can have a big impact on the
        size of the embedding matrix.

        A side effect of this method is the conversion from a quantized storage to an
        array storage.

        Returns
        -------
        embeddings : Embeddings
            Embeddings with an ExplicitVocab instead of a hash-based vocabulary.

        Raises
        ------
        TypeError
            If the current vocabulary is not a hash-based vocabulary
            (FinalfusionBucketVocab or FastTextVocab)
        """
        bucket_vocabs = (FastTextVocab, FinalfusionBucketVocab)
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
        return Embeddings(vocab=vocab, storage=NdArray(storage))

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

    Parameters
    ----------
    path : str
        Path to a file with embeddings in finalfusoin format.
    mmap : bool
        Toggles memory mapping the storage buffer.

    Returns
    -------
    embeddings : Embeddings
        The embeddings from the input file.
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
                embeddings.norms = Norms.read_chunk(file)
            elif chunk_id == ChunkIdentifier.Metadata:
                embeddings.metadata = Metadata.read_chunk(file)
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
    Read embeddings in word2vec binary format.

    Files are expected to start with a line containing rows and cols in utf-8. Words are encoded
    in utf-8 followed by a single whitespace. After the whitespace the embedding components are
    expected as little-endian float32.

    Parameters
    ----------
    path : str
        Path to a file with embeddings in word2vec binary format.

    Returns
    -------
    embeddings : Embeddings
        The embeddings from the input file.
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
    return Embeddings(storage=NdArray(matrix),
                      norms=Norms(norms),
                      vocab=SimpleVocab(words))


def load_textdims(path: str) -> Embeddings:
    """
    Read emebddings in textdims format.

    The first line contains whitespace separated rows and cols, the rest of the file contains
    whitespace separated word and vector components.

    Parameters
    ----------
    path : str
        Path to a file with embeddings in word2vec binary format.

    Returns
    -------
    embeddings : Embeddings
        The embeddings from the input file.
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
    return Embeddings(storage=NdArray(matrix),
                      norms=Norms(norms),
                      vocab=SimpleVocab(words))


def load_text(path: str) -> Embeddings:
    """
    Read embeddings in text format.

    Parameters
    ----------
    path : str
        Path to a file with embeddings in word2vec binary format.

    Returns
    -------
    embeddings : Embeddings
        Embeddings from the input file. The resulting Embeddings will have a
        SimpleVocab, NdArray and Norms.
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
    return Embeddings(storage=NdArray(matrix),
                      norms=Norms(norms),
                      vocab=SimpleVocab(words))


def load_fastText(path: str) -> Embeddings:  # pylint: disable=invalid-name
    """
    Read embeddings from a file in fastText format.

    Parameters
    ----------
    path : str
        Path to a file with embeddings in word2vec binary format.

    Returns
    -------
    embeddings : Embeddings
        The embeddings from the input file.
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
        storage = NdArray(matrix)
        norms = Norms(norms)
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
    metadata = Metadata({
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
    indexer = FastTextIndexer(buckets, min_n, max_n)
    return FastTextVocab(words, indexer)


_VOCAB_READERS = {
    ChunkIdentifier.SimpleVocab: SimpleVocab.read_chunk,
    ChunkIdentifier.BucketSubwordVocab: FinalfusionBucketVocab.read_chunk,
    ChunkIdentifier.FastTextSubwordVocab: FastTextVocab.read_chunk,
    ChunkIdentifier.ExplicitSubwordVocab: ExplicitVocab.read_chunk,
}

_STORAGE_READERS = {
    ChunkIdentifier.NdArray: NdArray.load,
    ChunkIdentifier.QuantizedArray: QuantizedArray.load,
}

__all__ = [
    'Embeddings', 'load_finalfusion', 'load_fastText', 'load_word2vec',
    'load_textdims', 'load_text'
]
