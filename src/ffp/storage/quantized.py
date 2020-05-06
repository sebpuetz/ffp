"""
Quantized finalfusion storage

This module contains the QuantizedArray storage type and the PQ quantizer.
Quantized storages offer a memory-for-speed trade-off and drastically reduce
the size of embedding matrices.
"""

import struct
from typing import IO, Tuple, Optional

import numpy as np

import ffp.io
from ffp.storage.storage import Storage


class QuantizedArray(Storage):
    """
    QuantizedArray storage.

    The QuantizedArray storage wraps a numpy array of quantized embeddings,
    optionally corresponding norms and a product quantizer.
    """
    def __init__(self, pq, quantized_embeddings, norms):
        super().__init__()
        self._quantizer = pq
        self._quantized_embeddings = quantized_embeddings
        self._norms = norms

    @property
    def shape(self) -> Tuple[int, int]:
        return self._quantized_embeddings.shape[
            0], self._quantizer.reconstructed_len

    def __getitem__(self, key):
        if key is None:
            raise TypeError("None is not a valid key.")
        quantized = self._quantized_embeddings[key]
        if isinstance(key, list):
            return self._quantizer.reconstruct_batch(quantized)
        return self._quantizer.reconstruct_vector(quantized)

    def __iter__(self):
        return map(self._quantizer.reconstruct_vector,
                   self._quantized_embeddings)

    @property
    def quantized_len(self) -> int:
        """
        Returns the length of the quantized embeddings.
        """
        return self._quantized_embeddings.shape[1]

    @staticmethod
    def chunk_identifier() -> ffp.io.ChunkIdentifier:
        return ffp.io.ChunkIdentifier.QuantizedArray

    @staticmethod
    def _read_quantized_header(file: IO[bytes]):
        projection = struct.unpack("<I", file.read(4))[0] != 0
        read_norms = struct.unpack("<I", file.read(4))[0] != 0
        quantized_len = struct.unpack("<I", file.read(4))[0]
        reconstructed_len = struct.unpack("<I", file.read(4))[0]
        n_centroids = struct.unpack("<I", file.read(4))[0]
        n_embeddings = struct.unpack("<Q", file.read(8))[0]
        assert reconstructed_len % quantized_len == 0
        type_id = ffp.io.TypeId(
            struct.unpack("<I", file.read(struct.calcsize("<I")))[0])
        ffp.io.TypeId.u8.match(type_id)
        type_id = ffp.io.TypeId(
            struct.unpack("<I", file.read(struct.calcsize("<I")))[0])
        ffp.io.TypeId.f32.match(type_id)
        file.seek(ffp.io.pad_float(file.tell()), 1)
        if projection:
            projection = np.fromfile(file,
                                     count=reconstructed_len *
                                     reconstructed_len,
                                     dtype=np.float32)
            projection = projection.reshape(
                (reconstructed_len, reconstructed_len))
        else:
            projection = None
        quantizer_shape = (quantized_len, n_centroids,
                           reconstructed_len // quantized_len)
        quantizers = np.fromfile(file,
                                 count=quantized_len * n_centroids *
                                 (reconstructed_len // quantized_len),
                                 dtype=np.float32)
        quantizers = quantizers.reshape(quantizer_shape)
        if read_norms:
            norms = np.fromfile(file, count=n_embeddings, dtype=np.float32)
        else:
            norms = None
        quantizer = PQ(quantizers, projection)
        return quantizer, (n_embeddings, quantized_len), norms

    @staticmethod
    def read_chunk(file) -> 'QuantizedArray':
        quantizer, embeds_shape, norms = QuantizedArray._read_quantized_header(
            file)
        n_embeddings, quantized_len = embeds_shape
        quantized_embeddings = np.fromfile(file,
                                           count=n_embeddings * quantized_len,
                                           dtype=np.uint8)
        quantized_embeddings = quantized_embeddings.reshape(embeds_shape)
        return QuantizedArray(quantizer, quantized_embeddings, norms)

    @staticmethod
    def mmap_chunk(file) -> 'QuantizedArray':
        quantizer, embeds_shape, norms = QuantizedArray._read_quantized_header(
            file)
        n_embeddings, quantized_len = embeds_shape
        offset = file.tell()
        file.seek(n_embeddings * quantized_len, 1)
        quantized_embeddings = np.memmap(file.name,
                                         dtype=np.uint8,
                                         mode='r',
                                         offset=offset,
                                         shape=embeds_shape)
        return QuantizedArray(quantizer, quantized_embeddings, norms)

    def write_chunk(self, file: IO[bytes]):
        file.write(struct.pack("<I", int(self.chunk_identifier())))
        padding = ffp.io.pad_float(file.tell())
        chunk_len = struct.calcsize("<IIIIIQII") + padding
        proj = self._quantizer.projection is not None
        if proj:
            chunk_len += struct.calcsize("<f") * pow(
                self._quantizer.reconstructed_len, 2)
        chunk_len += struct.calcsize("<f") * self._quantizer.subquantizers.size
        norms = self._norms is not None
        if norms:
            chunk_len += struct.calcsize("<f") * self._norms.size
        chunk_len += self._quantized_embeddings.size
        file.write(struct.pack("<Q", chunk_len))
        file.write(
            struct.pack("<IIIII", proj, norms, self.quantized_len,
                        self.shape[1], self._quantizer.n_centroids))
        file.write(struct.pack("<Q", self.shape[0]))
        file.write(
            struct.pack("<II", int(ffp.io.TypeId.u8), int(ffp.io.TypeId.f32)))
        file.write(struct.pack("x" * padding))
        if proj:
            file.write(self._quantizer.projection.tobytes())
        file.write(self._quantizer.subquantizers.tobytes())
        if norms:
            file.write(self._norms.tobytes())
        file.write(self._quantized_embeddings.tobytes())


class PQ:
    """
    Product Quantizer
    """
    def __init__(self, quantizers, projection):
        self._quantizers = quantizers
        self._reconstructed_len = quantizers.shape[0] * quantizers.shape[2]
        if projection is not None:
            assert projection.shape[
                0] == self._reconstructed_len == projection.shape[1]
        self._projection = projection

    @property
    def n_centroids(self) -> int:
        """
        Number of centroids per subquantizer.
        @return: number of centroids
        """
        return self._quantizers.shape[1]

    @property
    def projection(self) -> Optional[np.ndarray]:
        """
        Projection matrix
        @return: None or numpy.ndarray of shape reconstructed_len^2
        """
        return self._projection

    @property
    def reconstructed_len(self) -> int:
        """
        Reconstructed length
        @return: Length of reconstructed embeddings
        """
        return self._reconstructed_len

    @property
    def subquantizers(self) -> np.ndarray:
        """
        Quantizers tensor of shape:
        `quantizers * n_centroids * reconstructed_len / quantizers`
        @return: 3d tensor of quantizers
        """
        return self._quantizers

    def reconstruct_batch(self, quantized) -> np.ndarray:
        """
        Reconstruct a batch of vectors.
        @param quantized: Matrix of quantized embeddings.
        @return: Matrix of reconstructed embeddings.
        """
        reconstructed = np.zeros((quantized.shape[0], self.reconstructed_len),
                                 dtype=np.float32)
        for i in range(reconstructed.shape[0]):  # pylint: disable=unsubscriptable-object
            self.reconstruct_vector(quantized[i], reconstructed[i, :])
        return reconstructed

    def reconstruct_vector(self, quantized, out=None) -> np.ndarray:
        """
        Reconstruct a single vector.
        @param quantized: Quantized embedding vector.
        @param out: Optional array to write the reconstructed vector to.
        @return: the reconstructed vector.
        """
        if out is None:
            out = self._quantizers[np.arange(self._quantizers.shape[0]
                                             ), quantized].reshape(
                                                 self._reconstructed_len)
        else:
            out[:] = self._quantizers[np.arange(self._quantizers.shape[0]
                                                ), quantized].reshape(
                                                    self._reconstructed_len)
        if self.projection is not None:
            out.dot(self._projection.T, out=out)
        return out
