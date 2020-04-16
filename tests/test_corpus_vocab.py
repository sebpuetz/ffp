import os

import ffp
import tempfile


def test_fasttext_from_corpus_roundtrip(tests_root):
    path = os.path.join(tests_root, "data", "test.txt")
    vocab, _ = ffp.vocab.FastTextVocab.from_corpus(path)
    vocab.write(
        os.path.join(tempfile.gettempdir(), "fasttext_from_corpus.fifu"))
    vocab2 = ffp.vocab.load_vocab(
        os.path.join(tempfile.gettempdir(), "fasttext_from_corpus.fifu"))
    assert vocab == vocab2


def test_fasttext_from_corpus_minfreq(tests_root):
    path = os.path.join(tests_root, "data", "test.txt")
    cutoff = ffp.vocab.Cutoff(1, 'min_freq')

    vocab, token_counts = ffp.vocab.FastTextVocab.from_corpus(path, cutoff)
    assert len(vocab) == len(token_counts)
    assert vocab.idx_bound == len(vocab) + vocab.indexer.idx_bound
    assert vocab.words == ["test", "and", "a", "with", "random", "lines"]
    assert [token_counts[vocab[word]] for word in vocab] == [4, 3, 2] + 3 * [1]
    assert vocab.indexer("A") == 1118412


def test_fasttext_from_corpus_target_size(tests_root):
    path = os.path.join(tests_root, "data", "test.txt")
    indexer = ffp.subwords.FastTextIndexer()
    cutoff = ffp.vocab.Cutoff(0, 'target_size')
    vocab, token_counts = ffp.vocab.FastTextVocab.from_corpus(path, cutoff)
    assert len(vocab) == len(token_counts) == 0
    assert vocab.words == []
    assert vocab.idx_bound == indexer.idx_bound

    cutoff.cutoff = 1
    vocab, token_counts = ffp.vocab.FastTextVocab.from_corpus(path, cutoff)
    assert len(vocab) == len(token_counts) == 1
    assert vocab.idx_bound == indexer.idx_bound + len(vocab)
    assert vocab.words == ["test"]
    assert token_counts[vocab["test"]] == 4

    cutoff.cutoff = 3
    vocab, token_counts = ffp.vocab.FastTextVocab.from_corpus(path, cutoff)
    assert vocab.words == ["test", "and", "a"]
    assert len(vocab) == len(token_counts)
    assert vocab.idx_bound == indexer.idx_bound + len(vocab)


def test_simple_from_corpus_minfreq(tests_root):
    path = os.path.join(tests_root, "data", "test.txt")
    cutoff = ffp.vocab.Cutoff(1, 'min_freq')
    vocab, token_counts = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    assert len(vocab) == len(token_counts) == vocab.idx_bound == 6
    words = vocab.words
    assert words == ["test", "and", "a", "with", "random", "lines"]
    assert [token_counts[vocab[word]] for word in words] == [4, 3, 2] + 3 * [1]


def test_simple_from_corpus_target_size(tests_root):
    cutoff = ffp.vocab.Cutoff(0, 'target_size')
    path = os.path.join(tests_root, "data", "test.txt")
    vocab, token_counts = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    assert len(vocab) == len(token_counts) == 0
    assert vocab.words == []
    assert vocab.idx_bound == 0

    cutoff.cutoff = 1
    vocab, token_counts = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    assert len(vocab) == len(token_counts) == vocab.idx_bound == 1
    assert vocab.words == ["test"]
    assert token_counts[vocab["test"]] == 4

    cutoff.cutoff = 3
    vocab, token_counts = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    assert vocab.words == ["test", "and", "a"]
    assert len(vocab) == len(token_counts) == vocab.idx_bound == 3

    cutoff.cutoff = 4
    vocab, token_counts = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    assert vocab.words == ["test", "and", "a"]
    assert len(vocab) == len(token_counts) == vocab.idx_bound == 3

    cutoff.cutoff = 5
    vocab, token_counts = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    assert vocab.words == ["test", "and", "a"]
    assert len(vocab) == len(token_counts) == vocab.idx_bound == 3

    cutoff.cutoff = 6
    vocab, token_counts = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    assert vocab.words == ["test", "and", "a", "with", "random", "lines"]
    assert len(vocab) == len(token_counts) == vocab.idx_bound == 6

    cutoff.cutoff = 15
    vocab, token_counts = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    assert vocab.words == ["test", "and", "a", "with", "random", "lines"]
    assert len(vocab) == len(token_counts) == vocab.idx_bound == 6


def test_simple_from_corpus_roundtrip(tests_root):
    path = os.path.join(tests_root, "data", "test.txt")
    cutoff = ffp.vocab.Cutoff(1, 'min_freq')
    vocab, _ = ffp.vocab.SimpleVocab.from_corpus(path, cutoff)
    vocab.write(
        os.path.join(tempfile.gettempdir(), "explicit_from_corpus.fifu"))
    vocab2 = ffp.vocab.load_vocab(
        os.path.join(tempfile.gettempdir(), "explicit_from_corpus.fifu"))
    assert vocab == vocab2


def test_explicit_from_corpus_minfreq(tests_root):
    path = os.path.join(tests_root, "data", "test.txt")
    token_cutoff = ffp.vocab.Cutoff(1, 'min_freq')
    ngram_cutoff = ffp.vocab.Cutoff(1, 'min_freq')
    vocab, token_counts, ngram_counts = ffp.vocab.ExplicitVocab.from_corpus(
        path, (3, 3), token_cutoff, ngram_cutoff)
    assert len(vocab) == len(token_counts) == 6
    assert len(ngram_counts) == 22
    assert vocab.idx_bound == len(token_counts) + len(ngram_counts)

    words = vocab.words
    assert words == ["test", "and", "a", "with", "random", "lines"]
    assert [token_counts[vocab[word]] for word in words] == [4, 3, 2] + 3 * [1]
    ngrams = vocab.indexer.ngrams
    assert ngrams == [
        "tes", "st>", "est", "and", "<te", "nd>", "<an", "<a>", "wit", "th>",
        "ran", "om>", "nes", "ndo", "lin", "ith", "ine", "es>", "dom", "<wi",
        "<ra", "<li"
    ]
    for (idx, ngram) in enumerate(ngrams):
        assert vocab.indexer(ngram) == idx
    assert [ngram_counts[vocab.indexer(ngram)]
            for ngram in ngrams] == [4] * 5 + [3] * 2 + [2] + [1] * 14


def test_explicit_from_corpus_target_size(tests_root):
    path = os.path.join(tests_root, "data", "test.txt")
    token_cutoff = ffp.vocab.Cutoff(0, 'target_size')
    ngram_cutoff = ffp.vocab.Cutoff(1, 'min_freq')
    vocab, token_counts, ngram_counts = ffp.vocab.ExplicitVocab.from_corpus(
        path, (3, 3), token_cutoff, ngram_cutoff)
    assert len(vocab) == len(token_counts) == 0
    assert vocab.words == []
    ngrams = vocab.indexer.ngrams
    assert ngrams == [
        "tes", "st>", "est", "and", "<te", "nd>", "<an", "<a>", "wit", "th>",
        "ran", "om>", "nes", "ndo", "lin", "ith", "ine", "es>", "dom", "<wi",
        "<ra", "<li"
    ]
    for (idx, ngram) in enumerate(ngrams):
        assert vocab.indexer(ngram) == idx
    assert vocab.idx_bound == len(ngram_counts) == 22
    token_cutoff.cutoff = 1
    vocab, token_counts, ngram_counts = ffp.vocab.ExplicitVocab.from_corpus(
        path, (3, 3), token_cutoff, ngram_cutoff)
    assert len(vocab) == len(token_counts) == 1
    assert vocab.idx_bound == len(vocab) + len(ngram_counts) == 23
    assert vocab.words[0] == "test"
    assert token_counts[vocab["test"]] == 4

    ngram_cutoff.cutoff = 7
    ngram_cutoff.mode = "target_size"
    vocab, token_counts, ngram_counts = ffp.vocab.ExplicitVocab.from_corpus(
        path, (3, 3), token_cutoff, ngram_cutoff)
    ngrams = vocab.indexer.ngrams
    assert ngrams == ["tes", "st>", "est", "and", "<te", "nd>", "<an"]
    assert len(ngram_counts) == 7
    assert len(vocab) == len(token_counts) == 1
    assert vocab.idx_bound == len(vocab) + len(ngram_counts)
    for (idx, ngram) in enumerate(ngrams):
        assert vocab.indexer(ngram) == idx


def test_explicit_from_corpus_roundtrip(tests_root):
    path = os.path.join(tests_root, "data", "test.txt")
    cutoff = ffp.vocab.Cutoff(1, mode='min_freq')
    vocab, _, _ = ffp.vocab.ExplicitVocab.from_corpus(path,
                                                      token_cutoff=cutoff,
                                                      ngram_cutoff=cutoff)
    vocab.write(
        os.path.join(tempfile.gettempdir(), "explicit_from_corpus.fifu"))
    vocab2 = ffp.vocab.load_vocab(
        os.path.join(tempfile.gettempdir(), "explicit_from_corpus.fifu"))
    assert vocab == vocab2
