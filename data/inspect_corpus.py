# import sys
# from pathlib import Path
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))
# from tokenizer.tokenizer import GenieeTokenizer
# from data.build_dataset import GenieeTextDataset


# # ============================================================
# # CONFIGURATION
# # ============================================================

# CORPUS_DIR = Path("corpus")
# TOKENIZER_PATH = Path("tokenizer/artifacts/geniee.model")

# MAX_SEQ_LEN = 128


# # ============================================================
# # LOAD CORPUS
# # ============================================================

# def load_corpus():

#     text_files = sorted(CORPUS_DIR.rglob("*.txt"))

#     if not text_files:
#         raise RuntimeError(
#             f"No .txt files found inside: {CORPUS_DIR.resolve()}"
#         )

#     print("=" * 70)
#     print("GENIEE CORPUS INSPECTION")
#     print("=" * 70)

#     total_chars = 0
#     total_words = 0

#     print("\nCorpus files:\n")

#     for file_path in text_files:

#         text = file_path.read_text(
#             encoding="utf-8"
#         ).strip()

#         chars = len(text)
#         words = len(text.split())

#         total_chars += chars
#         total_words += words

#         print(
#             f"✓ {file_path} "
#             f"| chars={chars:,} "
#             f"| words={words:,}"
#         )

#     print("\n" + "-" * 70)

#     print(f"Files        : {len(text_files)}")
#     print(f"Characters   : {total_chars:,}")
#     print(f"Words        : {total_words:,}")

#     return text_files


# # ============================================================
# # LOAD TOKENIZER
# # ============================================================

# def load_tokenizer():

#     print("\nLoading tokenizer...")

#     tokenizer = GenieeTokenizer(
#         model_path=str(TOKENIZER_PATH)
#     )

#     print("Tokenizer loaded successfully.")

#     print(f"Vocabulary size : {tokenizer.vocab_size}")
#     print(f"UNK id          : {tokenizer.unk_id}")
#     print(f"BOS id          : {tokenizer.bos_id}")
#     print(f"EOS id          : {tokenizer.eos_id}")

#     return tokenizer


# # ============================================================
# # BUILD COMBINED TEXT
# # ============================================================

# def build_combined_text(text_files):

#     parts = []

#     for file_path in text_files:

#         text = file_path.read_text(
#             encoding="utf-8"
#         ).strip()

#         if not text:
#             continue

#         parts.append(text)

#     return "\n\n".join(parts)


# # ============================================================
# # TOKEN STATISTICS
# # ============================================================

# def inspect_tokens(tokenizer, text):

#     print("\n" + "=" * 70)
#     print("TOKEN STATISTICS")
#     print("=" * 70)

#     token_ids = tokenizer.encode(text)

#     print(f"\nTotal tokens : {len(token_ids):,}")

#     unique_tokens = len(set(token_ids))

#     print(f"Unique tokens: {unique_tokens:,}")

#     unk_count = token_ids.count(tokenizer.unk_id)

#     print(f"UNK tokens   : {unk_count:,}")

#     if len(token_ids) > 0:

#         unk_percentage = (
#             unk_count / len(token_ids)
#         ) * 100

#         print(
#             f"UNK percentage: "
#             f"{unk_percentage:.2f}%"
#         )

#     return token_ids


# # ============================================================
# # DATASET INSPECTION
# # ============================================================

# def inspect_dataset(token_ids):

#     print("\n" + "=" * 70)
#     print("TRAINING DATASET")
#     print("=" * 70)

#     dataset = GenieeTextDataset(
#         token_ids=token_ids,
#         max_seq_len=MAX_SEQ_LEN
#     )

#     print(
#         f"\nMax sequence length : {MAX_SEQ_LEN}"
#     )

#     print(
#         f"Training sequences  : {len(dataset):,}"
#     )

#     return dataset


# # ============================================================
# # SHOW TRAINING SAMPLES
# # ============================================================

# def show_samples(
#     tokenizer,
#     dataset,
#     number_of_samples=5
# ):

#     print("\n" + "=" * 70)
#     print("TRAINING SAMPLE INSPECTION")
#     print("=" * 70)

#     total = min(
#         number_of_samples,
#         len(dataset)
#     )

#     for index in range(total):

#         input_ids, target_ids = dataset[index]

#         input_list = input_ids.tolist()
#         target_list = target_ids.tolist()

#         decoded_input = tokenizer.decode(
#             input_list
#         )

#         decoded_target = tokenizer.decode(
#             target_list
#         )

#         print("\n")
#         print("-" * 70)

#         print(f"SAMPLE {index + 1}")

#         print("\nInput IDs:")
#         print(input_list)

#         print("\nTarget IDs:")
#         print(target_list)

#         print("\nDecoded Input:")
#         print(decoded_input)

#         print("\nDecoded Target:")
#         print(decoded_target)


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     text_files = load_corpus()

#     tokenizer = load_tokenizer()

#     combined_text = build_combined_text(
#         text_files
#     )

#     token_ids = inspect_tokens(
#         tokenizer,
#         combined_text
#     )

#     dataset = inspect_dataset(
#         token_ids
#     )

#     show_samples(
#         tokenizer,
#         dataset,
#         number_of_samples=5
#     )

#     print("\n" + "=" * 70)
#     print("INSPECTION COMPLETE")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()
from collections import Counter
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.corpus import GenieeCorpus


def main():

    print()
    print("=" * 60)
    print("GENIEE CORPUS INSPECTION")
    print("=" * 60)


    corpus = GenieeCorpus(
        corpus_dir="corpus",
        validation_split=0.1,
        seed=42
    )


    documents = corpus.load_documents()


    print()
    print(
        f"Total documents : {len(documents)}"
    )


    # ----------------------------------------------------------
    # Category statistics
    # ----------------------------------------------------------

    category_counts = Counter(
        document["category"]
        for document in documents
    )


    print()
    print("Documents by category:")
    print()


    for category, count in sorted(
        category_counts.items()
    ):

        print(
            f"  {category:<20} : {count}"
        )


    # ----------------------------------------------------------
    # Character statistics
    # ----------------------------------------------------------

    total_characters = sum(
        len(document["text"])
        for document in documents
    )


    total_words = sum(
        len(
            document["text"].split()
        )
        for document in documents
    )


    print()
    print(
        f"Total characters : {total_characters:,}"
    )

    print(
        f"Total words      : {total_words:,}"
    )


    # ----------------------------------------------------------
    # Instruction documents
    # ----------------------------------------------------------

    instruction_documents = [
        document
        for document in documents
        if document["category"] == "instructions"
    ]


    print()
    print(
        "Instruction documents:"
    )


    for document in instruction_documents:

        word_count = len(
            document["text"].split()
        )


        print(
            f"  {document['path']}"
            f" -> {word_count:,} words"
        )


    print()
    print("=" * 60)
    print("CORPUS INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()