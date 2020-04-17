# ffp

Interface to [finalfusion](https://finalfusion.github.io) written in (almost) pure Python. `ffp` supports reading from various embedding formats and more liberal construction of embeddings from components compared to the other `finalfusion` interfaces. Lots of pretrained [finalfrontier](https://github.com/finalfusion/finalfrontier/) embeddings are available [here](https://finalfusion.github.io/pretrained), fastText embeddings converted to finalfusion can be found [here](http://www.sfs.uni-tuebingen.de/a3-public-data/finalfusion-fasttext/)  

This is an early version of `ffp`, feedback is very much appreciated!

## Features

`ffp` supports most widely used embedding formats, currently not supported are fastText embeddings, those can be converted to the compatible `finalfusion` format through [finalfusion-utils](https://github.com/finalfusion/finalfusion-utils). Apart from  quantized storage, all `finalfusion` chunks can be read.

`ffp` provides construction, reading and writing of single `finalfusion` chunks, i.e., vocabularies, storage, norms, etc. can be read from or written to a `finalfusion` file in any combination. There are no assumptions about what constitutes a `finalfusion` file other than having at least a single chunk.

`ffp` integrates directly with `numpy` as the `NdArray` storage is a subclass of `numpy.ndarray`. All common numpy operations are available for this storage type.

Currently supported file formats:
* finalfusion
* text(-dims)
* word2vec binary

Currently supported Chunks:
* NdArray (mmap, in-memory)
* all vocabulary types
* Metadata
* Norms

## How to...

* ...install:
   - from pypi:
    ~~~Bash
    pip install ffp
    ~~~
   - from source:
    ~~~Bash
    git clone https://github.com/sebpuetz/ffp
    cd ffp
    pip install cython
    python setup.py install
    ~~~

* ...read embeddings from a file in finalfusion format and query for an embedding:
~~~Python
import ffp
import numpy as np

embeddings = ffp.load_finalfusion("path/to/file.fifu", "finalfusion")
res = embeddings["Query"]
# reading into an output array
in_vocab_embeddings = np.zeros((len(embeddings.vocab), embeddings.storage.shape[1]))
for word in embeddings.vocab:
    # Embeddings.embedding also returns `out`
    out = embeddings.embedding(word, out=in_vocab_embeddings[i])
    assert np.allclose(in_vocab_embeddings[i], out) 
~~~

* ...read the vocabulary from a file in `finalfusion` format:
~~~Python
import ffp

vocab = ffp.vocab.load_vocab("path/to/file.fifu")
~~~

* ...construct an `ExplicitVocab` from a corpus and write it to a file:
~~~Python
import ffp

# discard all ngrams appearing less than 30 times in the corpus
ngram_cutoff = ffp.vocab.Cutoff(30, "min_freq")
# keep less than 500,000 tokens in the vocabulary, setting the cutoff at the next frequency boundary
token_cutoff = ffp.vocab.Cutoff(500000, "target_size")
# extract ngrams in range 3 to 6 (including 6)
ngram_range = (3, 6)
vocab, token_counts, ngram_counts = ffp.vocab.ExplicitVocab.from_corpus("whitespace-tokenized-corpus.txt", ngram_range, token_cutoff, ngram_cutoff)
vocab.write("explicit_vocab.fifu")
~~~

* ...construct `Embeddings` with a `SimpleVocab` extracted from a corpus and a randomly initialized matrix:
~~~Python
import ffp
import numpy as np

# keep less than 500,000 tokens in the vocabulary, setting the cutoff at the next frequency boundary
token_cutoff = ffp.vocab.Cutoff(500000, "target_size")
vocab, _ = ffp.vocab.SimpleVocab.from_corpus("whitespace-tokenized-corpus.txt", token_cutoff)
rand_matrix = np.float32(np.random.rand(vocab.idx_bound, 300))
storage = ffp.storage.NdArray(rand_matrix)
e = ffp.embeddings.Embeddings(vocab=vocab, storage=storage)
~~~

**Note** This is not the [official python module](https://github.com/finalfusion/finalfusion-python) for finalfusion.
