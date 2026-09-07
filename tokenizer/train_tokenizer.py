# from pathlib import Path

# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.trainers import BpeTrainer


# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CORPUS_FILE = PROJECT_ROOT / "data" / "raw" / "corpus.txt"

# TOKENIZER_DIR = PROJECT_ROOT / "data" / "processed" / "tokenizer"

# TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"


# VOCAB_SIZE = 32000


# SPECIAL_TOKENS = [
#     "<PAD>",
#     "<UNK>",
#     "<BOS>",
#     "<EOS>",
# ]


# def train_tokenizer():
#     if not CORPUS_FILE.exists():
#         raise FileNotFoundError(
#             f"Training corpus not found: {CORPUS_FILE}"
#         )

#     TOKENIZER_DIR.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     print("=" * 60)
#     print("GENIEE TOKENIZER TRAINING")
#     print("=" * 60)

#     print(f"Corpus : {CORPUS_FILE}")
#     print(f"Output : {TOKENIZER_FILE}")
#     print(f"Vocab  : {VOCAB_SIZE}")

#     tokenizer = Tokenizer(
#         BPE(
#             unk_token="<UNK>"
#         )
#     )

#     tokenizer.pre_tokenizer = Whitespace()

#     # trainer = BpeTrainer(
#     #     vocab_size=VOCAB_SIZE,
#     #     special_tokens=SPECIAL_TOKENS,
#     #     min_frequency=1,
#     # )
#     trainer = BpeTrainer(
#     vocab_size=VOCAB_SIZE,
#     special_tokens=SPECIAL_TOKENS,
#     min_frequency=1,
#     initial_alphabet=list(
#         "abcdefghijklmnopqrstuvwxyz"
#         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     ),)

#     tokenizer.train(
#         files=[str(CORPUS_FILE)],
#         trainer=trainer,
#     )

#     tokenizer.save(
#         str(TOKENIZER_FILE)
#     )

#     print()
#     print("Tokenizer training completed.")
#     print(f"Saved tokenizer to:")
#     print(TOKENIZER_FILE)


# if __name__ == "__main__":
#     train_tokenizer()
# from pathlib import Path

# import sentencepiece as spm


# # --------------------------------------------------
# # Paths
# # --------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CORPUS_FILE = (
#     PROJECT_ROOT
#     / "corpus"
#     / "sample.txt"
# )

# TOKENIZER_DIR = (
#     PROJECT_ROOT
#     / "tokenizer"
#     / "artifacts"
# )

# TOKENIZER_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )


# # --------------------------------------------------
# # Configuration
# # --------------------------------------------------

# VOCAB_SIZE = 256

# MODEL_PREFIX = (
#     TOKENIZER_DIR
#     / "geniee"
# )


# def train_tokenizer():

#     # --------------------------------------------------
#     # Validate corpus
#     # --------------------------------------------------

#     if not CORPUS_FILE.exists():

#         raise FileNotFoundError(
#             f"Corpus not found: {CORPUS_FILE}"
#         )

#     # --------------------------------------------------
#     # Train SentencePiece tokenizer
#     # --------------------------------------------------

#     spm.SentencePieceTrainer.train(

#         input=str(CORPUS_FILE),

#         model_prefix=str(MODEL_PREFIX),

#         vocab_size=VOCAB_SIZE,

#         model_type="bpe",

#         character_coverage=1.0,

#         pad_id=0,

#         unk_id=1,

#         bos_id=2,

#         eos_id=3,

#         pad_piece="<pad>",

#         unk_piece="<unk>",

#         bos_piece="<bos>",

#         eos_piece="<eos>",

#         normalization_rule_name="nmt_nfkc",

#         input_sentence_size=0,

#         shuffle_input_sentence=True
#     )

#     print()
#     print("Tokenizer training completed.")
#     print()
#     print("Model:")
#     print(
#         f"{MODEL_PREFIX}.model"
#     )

#     print()
#     print("Vocabulary:")
#     print(
#         f"{MODEL_PREFIX}.vocab"
#     )


# if __name__ == "__main__":

#     train_tokenizer()


from pathlib import Path

import sentencepiece as spm


# ============================================================
# PATHS
# ============================================================

CORPUS_DIR = Path("corpus")
OUTPUT_DIR = Path("tokenizer/artifacts")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COLLECT CORPUS
# ============================================================

text_files = sorted(
    CORPUS_DIR.rglob("*.txt")
)

if not text_files:
    raise RuntimeError(
        f"No .txt files found in {CORPUS_DIR.resolve()}"
    )


print("=" * 70)
print("GENIEE TOKENIZER TRAINING")
print("=" * 70)

print("\nCorpus files:")

for file_path in text_files:

    text = file_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        print(
            f"SKIP EMPTY: {file_path}"
        )
        continue

    print(
        f"✓ {file_path} "
        f"({len(text):,} chars)"
    )


# ============================================================
# CREATE COMBINED CORPUS
# ============================================================

combined_corpus = OUTPUT_DIR / "combined_corpus.txt"

with combined_corpus.open(
    "w",
    encoding="utf-8"
) as output:

    for file_path in text_files:

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        output.write(text)
        output.write("\n\n")


print("\nCombined corpus:")
print(combined_corpus.resolve())


# ============================================================
# TRAIN SENTENCEPIECE
# ============================================================

model_prefix = str(
    OUTPUT_DIR / "geniee"
)

print("\nTraining tokenizer...")
spm.SentencePieceTrainer.train(
    input=str(combined_corpus),

    model_prefix=model_prefix,

    vocab_size=2048,

    model_type="bpe",

    character_coverage=1.0,

    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_id=0,

    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    pad_piece="<pad>",

    user_defined_symbols=[
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
    ],

    max_sentence_length=10000,

    shuffle_input_sentence=True,
    input_sentence_size=0,

    train_extremely_large_corpus=False,

    hard_vocab_limit=False,
)

# spm.SentencePieceTrainer.train(

#     input=str(combined_corpus),

#     model_prefix=model_prefix,

#     vocab_size=2048,

#     model_type="bpe",

#     character_coverage=1.0,

#     unk_id=1,
#     bos_id=2,
#     eos_id=3,
#     pad_id=0,

#     unk_piece="<unk>",
#     bos_piece="<s>",
#     eos_piece="</s>",
#     pad_piece="<pad>",

#     max_sentence_length=10000,

#     shuffle_input_sentence=True,

#     input_sentence_size=0,

#     train_extremely_large_corpus=False,

#     hard_vocab_limit=False,

#     user_defined_symbols=[
#         "<|User|>:",
#         "<|Assistant|>:",
#         "<|Geniee|>:"
#     ]
# )


# print("\nTokenizer training completed.")

# print(
#     f"Model: {OUTPUT_DIR / 'geniee.model'}"
# )

# print(
#     f"Vocab: {OUTPUT_DIR / 'geniee.vocab'}"
# )

# print("=" * 70)