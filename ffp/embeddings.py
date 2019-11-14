"""
Finalfusion Embeddings
"""

from enum import Enum, unique
from typing import Optional, Union, Tuple

import numpy as np

import ffp.io
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


class Embeddings:
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
        if vocab is not None and storage is not None:
            assert storage.shape[0] == vocab.idx_bound
        if vocab is not None and norms is not None:
            assert len(vocab) == len(norms)
        if storage is not None and norms is not None:
            assert storage.shape[0] >= len(norms)
        self.storage = storage
        self.vocab = vocab
        self.norms = norms
        self.metadata = metadata

    def chunks(self):
        """
        Get the present chunks as a list.
        :return: ffp.io.Chunks
        """
        chunks = []
        if self.vocab is not None:
            chunks.append(self.vocab)
        if self.storage is not None:
            chunks.append(self.storage)
        if self.metadata is not None:
            chunks.append(self.metadata)
        if self.norms is not None:
            chunks.append(self.norms)
        return chunks

    @staticmethod
    def read(path: str, emb_format: Union[str, Format] = Format.finalfusion
             ) -> 'Embeddings':
        """
        Read embeddings.
        :param emb_format: the embeddings format, one of finalfusion, finalfusion_mmap, text,
        textdims and word2vec
        :param path: file
        :return: Embeddings
        """
        if isinstance(emb_format, str):
            emb_format = Format(emb_format)
        if emb_format == Format.finalfusion:
            return Embeddings._read_finalfusion(path, False)
        if emb_format == Format.finalfusion_mmap:
            return Embeddings._read_finalfusion(path, True)
        if emb_format == Format.text:
            return Embeddings._read_text(path)
        if emb_format == Format.textdims:
            return Embeddings._read_textdims(path)
        if emb_format == Format.word2vec:
            return Embeddings._read_word2vec(path)
        raise TypeError("Unknown format")

    def write(self, path: str):
        """
        Write the Embeddings in finalfusion format to the given path.
        :param path: path
        """
        with open(path, 'wb') as file:
            chunks = self.chunks()
            header = ffp.io.Header(
                [chunk.chunk_identifier() for chunk in chunks])
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
        idx = self.vocab.idx(word)
        res = self.storage[idx]
        if res.ndim == 1:
            if out is not None:
                out[:] = res
            else:
                out = res
        elif idx is None:
            return default
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
        if self.norms is None:
            raise TypeError("embeddings don't contain norms chunk")
        idx = self.vocab.idx(word)
        res = self.storage[idx]
        if res.ndim == 1:
            if out is not None:
                out[:] = res
            else:
                out = res
            return out, self.norms[idx]
        if idx is None:
            return default

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
        if not isinstance(self.vocab, ffp.vocab.BucketVocab):
            raise TypeError(
                "Only bucketed embeddings can be converted to explicit.")
        vocab = self.vocab.to_explicit()
        if self.storage is None:
            return Embeddings(vocab=vocab)
        storage = np.zeros((vocab.idx_bound, self.storage.shape[1]),
                           dtype=np.float32)
        storage[:len(vocab)] = self.storage[:len(vocab)]
        for ngram in vocab.indexer:
            storage[len(vocab) + vocab.indexer(ngram)] = self.storage[
                len(vocab) + self.vocab.indexer(ngram)]
        return Embeddings(vocab=vocab, storage=ffp.storage.NdArray(storage))

    @staticmethod
    def _read_finalfusion(path: str, mmap: bool) -> 'Embeddings':
        """
        Read embeddings from a file in finalfusion format.
        :param path: path
        :param mmap: whether to mmap the storage
        :return: Embeddings
        """
        with open(path, 'rb') as file:
            _ = ffp.io.Header.read_chunk(file)
            chunk_id, _ = ffp.io.read_chunk_header(file)
            embeddings = Embeddings()
            while True:
                if chunk_id == ffp.io.ChunkIdentifier.NdArray:
                    embeddings.storage = ffp.storage.NdArray.load(file, mmap)
                elif chunk_id == ffp.io.ChunkIdentifier.SimpleVocab:
                    embeddings.vocab = ffp.vocab.SimpleVocab.read_chunk(file)
                elif chunk_id == ffp.io.ChunkIdentifier.BucketSubwordVocab:
                    embeddings.vocab = ffp.vocab.FinalfusionBucketVocab.read_chunk(
                        file)
                elif chunk_id == ffp.io.ChunkIdentifier.FastTextSubwordVocab:
                    embeddings.vocab = ffp.vocab.FastTextVocab.read_chunk(file)
                elif chunk_id == ffp.io.ChunkIdentifier.ExplicitSubwordVocab:
                    embeddings.vocab = ffp.vocab.ExplicitVocab.read_chunk(file)
                elif chunk_id == ffp.io.ChunkIdentifier.NdNorms:
                    embeddings.norms = ffp.norms.Norms.read_chunk(file)
                elif chunk_id == ffp.io.ChunkIdentifier.Metadata:
                    embeddings.metadata = ffp.metadata.Metadata.read_chunk(
                        file)
                else:
                    chunk_id, _ = ffp.io.read_chunk_header(file)
                    raise IOError(str(chunk_id) + " is not yet supported.")
                chunk_header = ffp.io.read_chunk_header(file)
                if chunk_header is None:
                    break
                chunk_id, _ = chunk_header
            return embeddings

    @staticmethod
    def _read_textdims(path):
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

    @staticmethod
    def _read_text(path: str) -> 'Embeddings':
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

    @staticmethod
    def _read_word2vec(path: str) -> 'Embeddings':
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

    def __getitem__(self, item):
        idx = self.vocab[item]
        res = self.storage[idx]
        if res.ndim == 1:
            return res
        embed_sum = res.sum(axis=0)
        return embed_sum / np.linalg.norm(embed_sum)

    def __contains__(self, item):
        if self.vocab is None:
            raise TypeError("These embeddings don't contain a vocabulary")
        return item in self.vocab

    def __repr__(self):
        return "Embeddings { \n" + "\n".join(
            [repr(chunk) for chunk in self.chunks()]) + "\n}"

    def __iter__(self):
        if self.norms is not None:
            return zip(self.vocab.words, self.storage, self.norms)
        return zip(self.vocab.words, self.storage)
