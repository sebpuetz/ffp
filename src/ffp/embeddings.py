"""
Finalfusion Embeddings
"""
from os import PathLike
from typing import Optional, Union, Tuple, List, BinaryIO

import numpy as np

from ffp.io import Chunk, ChunkIdentifier, Header, _read_binary, _read_chunk_header,\
    FinalfusionFormatError
from ffp.metadata import Metadata
from ffp.norms import Norms
from ffp.storage import Storage, NdArray, QuantizedArray
from ffp.subwords import FastTextIndexer
from ffp.vocab import Vocab, FastTextVocab, FinalfusionBucketVocab, SimpleVocab, ExplicitVocab


class Embeddings:  # pylint: disable=too-many-instance-attributes
    """
    Embeddings class.

    Embeddings always contain a :class:`~finalfusion.storage.storage.Storage` and
    :class:`~finalfusion.vocab.vocab.Vocab`. Optional chunks are
    :class:`~finalfusion.norms.Norms` corresponding to the embeddings of the in-vocab tokens and
    :class:`~finalfusion.metadata.Metadata`.

    Embeddings can be retrieved through three methods:

    1. :meth:`Embeddings.embedding` allows to provide a default value and returns
       this value if no embedding could be found.
    2. :meth:`Embeddings.__getitem__` retrieves an embedding for the query but
       raises an exception if it cannot retrieve an embedding.
    3. :meth:`Embeddings.embedding_with_norm` requires a :class:`~finalfusion.norms.Norms`
       chunk and returns an embedding together with the corresponding L2 norm.

    Embeddings are composed of the 4 chunk types:

    1. :class:`~ffp.storage.Storage`: either :class:`~ffp.storage.ndarray.NdArray` or
       :class:`~ffp.storage.quantized.QuantizedArray` *(required)*
    2. :class:`~ffp.vocab.Vocab`, one of :class:`~ffp.vocab.simple_vocab.SimpleVocab`,
       :class:`~ffp.vocab.subword.FinalfusionBucketVocab`,
       :class:`~ffp.vocab.subword.FastTextVocab` and :class:`~ffp.vocab.subword.ExplicitVocab`
        *(required)*
    3. :class:`~ffp.norms.Norms`
    4. :class:`~ffp.metadata.Metadata`

    Examples
    --------
    >>> storage = NdArray(np.float32(np.random.rand(2, 10)))
    >>> vocab = SimpleVocab(["Some", "words"])
    >>> metadata = Metadata({"Some": "value", "numerical": 0})
    >>> norms = Norms(np.float32(np.random.rand(2)))
    >>> embeddings = Embeddings(storage=storage, vocab=vocab, metadata=metadata, norms=norms)
    >>> embeddings.vocab.words
    ['Some', 'words']
    >>> np.allclose(embeddings["Some"], storage[0])
    True
    >>> try:
    ...     embeddings["oov"]
    ... except KeyError:
    ...     True
    True
    >>> _, n = embeddings.embedding_with_norm("Some")
    >>> np.isclose(n, norms[0])
    True
    >>> embeddings.metadata
    {'Some': 'value', 'numerical': 0}
    """
    def __init__(self,
                 storage: Storage,
                 vocab: Vocab,
                 norms: Optional[Norms] = None,
                 metadata: Optional[Metadata] = None):
        """
        Initialize Embeddings.

        Initializes Embeddings with the given chunks.

        :Conditions:
            The following conditions need to hold if the respective chunks are passed.

            * Chunks need to have the expected type.
            * ``vocab.idx_bound == storage.shape[0]``
            * ``len(vocab) == len(norms)``
            * ``len(norms) == len(vocab) and len(norms) >= storage.shape[0]``

        Parameters
        ----------
        storage : Storage
            Embeddings Storage.
        vocab : Vocab
            Embeddings Vocabulary.
        norms : Norms, optional
            Embeddings Norms.
        metadata : Metadata, optional
            Embeddings Metadata.

        Raises
        ------
        AssertionError
            If any of the conditions don't hold.
        """
        Embeddings._check_requirements(storage, vocab, norms, metadata)
        self._storage = storage
        self._vocab = vocab
        self._norms = norms
        self._metadata = metadata

    def __getitem__(self, item: str) -> np.ndarray:
        """
        Returns an embeddings.

        Parameters
        ----------
        item : str
            The query item.

        Returns
        -------
        embedding : numpy.ndarray
            The embedding.

        Raises
        ------
        KeyError
            If no embedding could be retrieved.

        See Also
        --------
        :func:`~Embeddings.embedding`
        :func:`~Embeddings.embedding_with_norm`
        """
        # no need to check for none since Vocab raises KeyError if it can't produce indices
        idx = self._vocab[item]
        return self._embedding(idx)[0]

    def embedding(self,
                  word: str,
                  out: Optional[np.ndarray] = None,
                  default: Optional[np.ndarray] = None
                  ) -> Optional[np.ndarray]:
        """
        Embedding lookup.

        Looks up the embedding for the input word.

        If an `out` array is specified, the embedding is written into the array.

        If it is not possible to retrieve an embedding for the input word, the `default`
        value is returned. This defaults to `None`. An embedding can not be retrieved if
        the vocabulary cannot provide an index for `word`.

        This method never fails. If you do not provide a default value, check the return value
        for None. ``out`` is left untouched if no embedding can be found and ``default`` is None.

        Parameters
        ----------
        word : str
            The query word.
        out : numpy.ndarray, optional
            Optional output array to write the embedding into.
        default: numpy.ndarray, optional
            Optional default value to return if no embedding can be retrieved. Defaults to None.

        Returns
        -------
        embedding : numpy.ndarray, optional
            The retrieved embedding or the default value.

        Examples
        --------
        >>> matrix = np.float32(np.random.rand(2, 10))
        >>> storage = NdArray(matrix)
        >>> vocab = SimpleVocab(["Some", "words"])
        >>> embeddings = Embeddings(storage=storage, vocab=vocab)
        >>> np.allclose(embeddings.embedding("Some"), matrix[0])
        True
        >>> # default value is None
        >>> embeddings.embedding("oov") is None
        True
        >>> # It's possible to specify a default value
        >>> default = embeddings.embedding("oov", default=storage[0])
        >>> np.allclose(default, storage[0])
        True
        >>> # Embeddings can be written to an output buffer.
        >>> out = np.zeros(10, dtype=np.float32)
        >>> out2 = embeddings.embedding("Some", out=out)
        >>> out is out2
        True
        >>> np.allclose(out, matrix[0])
        True

        See Also
        --------
        :func:`~Embeddings.embedding_with_norm`
        :func:`~Embeddings.__getitem__`
        """
        idx = self._vocab.idx(word)
        if idx is None:
            if out is not None and default is not None:
                out[:] = default
                return out
            return default
        return self._embedding(idx, out)[0]

    def embedding_with_norm(self,
                            word: str,
                            out: Optional[np.ndarray] = None,
                            default: Optional[Tuple[np.ndarray, float]] = None
                            ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Embedding lookup with norm.

        Looks up the embedding for the input word together with its norm.

        If an `out` array is specified, the embedding is written into the array.

        If it is not possible to retrieve an embedding for the input word, the `default`
        value is returned. This defaults to `None`. An embedding can not be retrieved if
        the vocabulary cannot provide an index for `word`.

        This method raises a TypeError if norms are not set.

        Parameters
        ----------
        word : str
            The query word.
        out : numpy.ndarray, optional
            Optional output array to write the embedding into.
        default: Tuple[numpy.ndarray, float], optional
            Optional default value to return if no embedding can be retrieved. Defaults to None.

        Returns
        -------
        (embedding, norm) : EmbeddingWithNorm, optional
            Tuple with the retrieved embedding or the default value at the first index and the
            norm at the second index.

        See Also
        --------
        :func:`~Embeddings.embedding`
        :func:`~Embeddings.__getitem__`
        """
        if self._norms is None:
            raise TypeError("embeddings don't contain norms chunk")
        idx = self._vocab.idx(word)
        if idx is None:
            if out is not None and default is not None:
                out[:] = default[0]
                return out, default[1]
            return default
        return self._embedding(idx, out)

    @property
    def storage(self) -> Optional[Storage]:
        """
        Get the :class:`~finalfusion.storage.storage.Storage`.

        Returns
        -------
        storage : Storage
            The embeddings storage.
        """
        return self._storage

    @property
    def vocab(self) -> Optional[Vocab]:
        """
        The :class:`~finalfusion.vocab.vocab.Vocab`.

        Returns
        -------
        vocab : Vocab
            The vocabulary
        """
        return self._vocab

    @property
    def norms(self) -> Optional[Norms]:
        """
        The :class:`~finalfusion.vocab.vocab.Norms`.

        :Getter: Returns None or the Norms.
        :Setter: Set the Norms.

        Returns
        -------
        norms : Norms, optional
            The Norms or None.

        Raises
        ------
        AssertionError
            if ``embeddings.storage.shape[0] < len(embeddings.norms)`` or
            ``len(embeddings.norms) != len(embeddings.vocab)``
        TypeError
            If ``norms`` is neither Norms nor None.
        """
        return self._norms

    @norms.setter
    def norms(self, norms: Optional[Norms]):
        if norms is None:
            self._norms = None
        else:
            Embeddings._norms_compat(self.storage, self.vocab, self.norms)
            self._norms = norms

    @property
    def metadata(self) -> Optional[Metadata]:
        """
        The :class:`~finalfusion.vocab.vocab.Metadata`.

        :Getter: Returns None or the Metadata.
        :Setter: Set the Metadata.

        Returns
        -------
        metadata : Metadata, optional
            The Metadata or None.

        Raises
        ------
        TypeError
            If ``metadata`` is neither Metadata nor None.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[Metadata]):
        if metadata is None:
            self._metadata = None
        elif isinstance(metadata, Metadata):
            self._metadata = metadata
        else:
            raise TypeError("Expected 'None' or 'Metadata'.")

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
        storage = np.zeros((vocab.idx_bound, self._storage.shape[1]),
                           dtype=np.float32)
        storage[:len(vocab)] = self._storage[:len(vocab)]
        for ngram in vocab.subword_indexer:
            storage[len(vocab) + vocab.subword_indexer(ngram)] = self._storage[
                len(vocab) + self._vocab.subword_indexer(ngram)]
        return Embeddings(vocab=vocab, storage=NdArray(storage))

    def chunks(self) -> List[Chunk]:
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
        if self._metadata is not None:
            chunks.append(self.metadata)
        chunks.append(self.vocab)
        chunks.append(self.storage)
        if self._norms is not None:
            chunks.append(self.norms)
        return chunks

    def write(self, file: str):
        """
        Write the Embeddings to the given file.

        Writes the Embeddings to a finalfusion file at the given file.

        Parameters
        ----------
        file : str
            Path of the output file.
        """
        with open(file, 'wb') as outf:
            chunks = self.chunks()
            header = Header([chunk.chunk_identifier() for chunk in chunks])
            header.write_chunk(outf)
            for chunk in chunks:
                chunk.write_chunk(outf)

    def __contains__(self, item):
        return item in self._vocab

    def __iter__(self):
        if self._norms is not None:
            return zip(self._vocab.words, self._storage, self._norms)
        return zip(self._vocab.words, self._storage)

    def _embedding(self,
                   idx: Union[int, List[int]],
                   out: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, Optional[float]]:
        res = self._storage[idx]
        if res.ndim == 1:
            if out is not None:
                out[:] = res
            else:
                out = res
            if self._norms is not None:
                norm = self._norms[idx]
            else:
                norm = None
        else:
            out = np.add.reduce(res, 0, out=out, keepdims=False)
            norm = np.linalg.norm(out)
            out /= norm
        return out, norm

    @staticmethod
    def _check_requirements(storage: Storage, vocab: Vocab,
                            norms: Optional[Norms],
                            metadata: Optional[Metadata]):
        assert isinstance(storage, Storage),\
            "storage is required to be a Storage"
        assert isinstance(vocab, Vocab), "vocab is required to be a Vocab"
        assert storage.shape[0] == vocab.idx_bound,\
            "Number of embeddings needs to be equal to vocab's idx_bound"
        if norms is not None:
            Embeddings._norms_compat(storage, vocab, norms)
        assert metadata is None or isinstance(metadata, Metadata),\
            "metadata is required to be Metadata"

    @staticmethod
    def _norms_compat(storage: Storage, vocab: Vocab, norms: Norms):
        assert isinstance(norms, Norms), "norms are required to be Norms"
        assert storage.shape[0] >= len(norms),\
            "Number of embeddings needs to be greater than or equal to number of norms."
        assert len(vocab) == len(norms),\
            "Vocab length needs to be equal to number of norms."


def load_finalfusion(file: Union[str, bytes, int, PathLike],
                     mmap: bool = False) -> Embeddings:
    """
    Read embeddings from a file in finalfusion format.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to a file with embeddings in finalfusoin format.
    mmap : bool
        Toggles memory mapping the storage buffer.

    Returns
    -------
    embeddings : Embeddings
        The embeddings from the input file.
    """
    with open(file, 'rb') as inf:
        _ = Header.read_chunk(inf)
        chunk_id, _ = _read_chunk_header(inf)
        norms = None
        metadata = None

        if chunk_id == ChunkIdentifier.Metadata:
            metadata = Metadata.read_chunk(inf)
            chunk_id, _ = _read_chunk_header(inf)

        if chunk_id.is_vocab():
            vocab = _VOCAB_READERS[chunk_id](inf)
        else:
            raise FinalfusionFormatError(
                f'Expected vocab chunk, not {str(chunk_id)}')

        chunk_id, _ = _read_chunk_header(inf)
        if chunk_id.is_storage():
            storage = _STORAGE_READERS[chunk_id](inf, mmap)
        else:
            raise FinalfusionFormatError(
                f'Expected vocab chunk, not {str(chunk_id)}')

        chunk_id = _read_chunk_header(inf)
        if chunk_id is not None:
            if chunk_id[0] == ChunkIdentifier.NdNorms:
                norms = Norms.read_chunk(inf)
            else:
                raise FinalfusionFormatError(
                    f'Expected vocab chunk, not {str(chunk_id)}')

        return Embeddings(storage=storage,
                          vocab=vocab,
                          norms=norms,
                          metadata=metadata)


def load_word2vec(file: Union[str, bytes, int, PathLike]) -> Embeddings:
    """
    Read embeddings in word2vec binary format.

    Files are expected to start with a line containing rows and cols in utf-8. Words are encoded
    in utf-8 followed by a single whitespace. After the whitespace the embedding components are
    expected as little-endian float32.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to a file with embeddings in word2vec binary format.

    Returns
    -------
    embeddings : Embeddings
        The embeddings from the input file.
    """
    words = []
    with open(file, 'rb') as inf:
        rows, cols = map(int, inf.readline().decode("utf-8").split())
        matrix = np.zeros((rows, cols), dtype=np.float32)
        for row in range(rows):
            word = []
            while True:
                byte = inf.read(1)
                if byte == b' ':
                    break
                if byte == b'':
                    raise EOFError
                if byte != b'\n':
                    word.append(byte)
            word = b''.join(word).decode('utf-8')
            words.append(word)
            vec = inf.read(cols * matrix.itemsize)
            matrix[row] = np.frombuffer(vec, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    matrix /= np.expand_dims(norms, axis=1)
    return Embeddings(storage=NdArray(matrix),
                      norms=Norms(norms),
                      vocab=SimpleVocab(words))


def load_textdims(file: Union[str, bytes, int, PathLike]) -> Embeddings:
    """
    Read emebddings in textdims format.

    The first line contains whitespace separated rows and cols, the rest of the file contains
    whitespace separated word and vector components.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to a file with embeddings in word2vec binary format.

    Returns
    -------
    embeddings : Embeddings
        The embeddings from the input file.
    """
    words = []
    with open(file) as inf:
        rows, cols = next(inf).split()
        matrix = np.zeros((int(rows), int(cols)), dtype=np.float32)
        for i, line in enumerate(inf):
            line = line.strip().split()
            words.append(line[0])
            matrix[i] = line[1:]
    norms = np.linalg.norm(matrix, axis=1)
    matrix /= np.expand_dims(norms, axis=1)
    return Embeddings(storage=NdArray(matrix),
                      norms=Norms(norms),
                      vocab=SimpleVocab(words))


def load_text(file: Union[str, bytes, int, PathLike]) -> Embeddings:
    """
    Read embeddings in text format.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to a file with embeddings in word2vec binary format.

    Returns
    -------
    embeddings : Embeddings
        Embeddings from the input file. The resulting Embeddings will have a
        SimpleVocab, NdArray and Norms.
    """
    words = []
    vecs = []
    with open(file) as inf:
        for line in inf:
            line = line.strip().split()
            words.append(line[0])
            vecs.append(line[1:])
        matrix = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    matrix /= np.expand_dims(norms, axis=1)
    return Embeddings(storage=NdArray(matrix),
                      norms=Norms(norms),
                      vocab=SimpleVocab(words))


def load_fastText(file: Union[str, bytes, int, PathLike]) -> Embeddings:  # pylint: disable=invalid-name
    """
    Read embeddings from a file in fastText format.

    Parameters
    ----------
    file : str, bytes, int, PathLike
        Path to a file with embeddings in word2vec binary format.

    Returns
    -------
    embeddings : Embeddings
        The embeddings from the input file.
    """
    with open(file, 'rb') as inf:
        _read_ft_header(inf)
        metadata = _read_ft_cfg(inf)
        vocab = _read_ft_vocab(inf, metadata['buckets'], metadata['min_n'],
                               metadata['max_n'])
        quantized = _read_binary(inf, "<B")[0]
        if quantized:
            raise NotImplementedError(
                "Quantized storage is not supported for fastText models")
        rows, cols = _read_binary(inf, "<QQ")
        matrix = np.fromfile(file=inf, count=rows * cols, dtype=np.float32)
        matrix = np.reshape(matrix, (rows, cols))
        for i, word in enumerate(vocab):
            indices = [i] + vocab.subword_indices(word)
            matrix[i] = matrix[indices].mean(0, keepdims=False)
        norms = np.linalg.norm(matrix[:len(vocab)], axis=1)
        matrix[:len(vocab)] /= np.expand_dims(norms, axis=1)
        storage = NdArray(matrix)
        norms = Norms(norms)
    return Embeddings(storage, vocab, norms, metadata)


def _read_ft_header(file: BinaryIO):
    magic, version = _read_binary(file, "<II")
    if magic != 793_712_314:
        raise ValueError(f"Magic should be 793_712_314, not: {magic}")
    if version > 12:
        raise ValueError(f"Expected version 12, not: {version}")


def _read_ft_cfg(file: BinaryIO) -> Metadata:
    cfg = _read_binary(file, "<12Id")
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


def _read_ft_vocab(file: BinaryIO, buckets: int, min_n: int,
                   max_n: int) -> FastTextVocab:
    vocab_size, _, n_labels = _read_binary(file, "<III")  # discard n_words
    if n_labels:
        raise NotImplementedError(
            "fastText prediction models are not supported")
    _, prune_idx_size = _read_binary(file, "<Qq")  # discard n_tokens
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
        _ = _read_binary(file, "<Q")  # discard frequency
        entry_type = _read_binary(file, "<B")[0]
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
