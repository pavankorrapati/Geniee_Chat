from pathlib import Path

from data.corpus import GenieeCorpus


def test_corpus_loads_text():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    corpus_dir = (
        project_root / "corpus"
    )

    corpus = GenieeCorpus(
        corpus_dir=corpus_dir
    )

    text = corpus.load_text()

    print()
    print("Corpus characters:", len(text))

    print()
    print("First 200 characters:")
    print(text[:200])

    assert isinstance(
        text,
        str
    )

    assert len(text) > 0


def test_corpus_finds_text_files():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    corpus_dir = (
        project_root / "corpus"
    )

    corpus = GenieeCorpus(
        corpus_dir=corpus_dir
    )

    files = corpus.get_text_files()

    print()
    print("Corpus files:")

    for file_path in files:
        print(file_path)

    assert len(files) > 0

    assert all(
        file_path.suffix == ".txt"
        for file_path in files
    )


def test_train_validation_split():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    corpus_dir = (
        project_root / "corpus"
    )

    corpus = GenieeCorpus(
        corpus_dir=corpus_dir,
        validation_split=0.1
    )

    train_text, validation_text = (
        corpus.train_validation_split()
    )

    print()
    print(
        "Training characters:",
        len(train_text)
    )

    print(
        "Validation characters:",
        len(validation_text)
    )

    assert len(train_text) > 0

    assert len(validation_text) > 0

    assert (
        len(train_text)
        + len(validation_text)
        == len(corpus.load_text())
    )