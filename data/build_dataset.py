# from pathlib import Path
# import sys
# import torch
# from torch.utils.data import DataLoader
# PROJECT_ROOT = (
#     Path(__file__).resolve().parent.parent
# )

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(
#         0,
#         str(PROJECT_ROOT)
#     )
# from data.corpus import GenieeCorpus
# from data.dataset import GenieeTextDataset
# from tokenizer.tokenizer import GenieeTokenizer


# class GenieeDataModule:
#     """
#     Builds the complete data pipeline for Geniee.

#     Pipeline:

#         Corpus
#           ↓
#         Text
#           ↓
#         Tokenizer
#           ↓
#         Token IDs
#           ↓
#         GenieeTextDataset
#           ↓
#         DataLoader
#     """

#     def __init__(
#         self,
#         corpus_dir,
#         tokenizer_path,
#         max_seq_len,
#         batch_size,
#         stride=None,
#         validation_split=0.1,
#         shuffle_train=True
#     ):

#         # --------------------------------------------------
#         # Store configuration
#         # --------------------------------------------------

#         self.corpus_dir = Path(corpus_dir)

#         self.tokenizer_path = Path(
#             tokenizer_path
#         )

#         self.max_seq_len = max_seq_len

#         self.batch_size = batch_size

#         self.stride = (
#             stride
#             if stride is not None
#             else max_seq_len
#         )

#         self.validation_split = (
#             validation_split
#         )

#         self.shuffle_train = (
#             shuffle_train
#         )

#         # --------------------------------------------------
#         # Create corpus
#         # --------------------------------------------------

#         self.corpus = GenieeCorpus(
#             corpus_dir=self.corpus_dir,
#             validation_split=validation_split
#         )

#         # --------------------------------------------------
#         # Create tokenizer
#         # --------------------------------------------------

#         self.tokenizer = GenieeTokenizer(
#             model_path=self.tokenizer_path
#         )

#         # --------------------------------------------------
#         # These will be populated by build()
#         # --------------------------------------------------

#         self.train_dataset = None

#         self.validation_dataset = None

#         self.train_loader = None

#         self.validation_loader = None

#         self.train_token_ids = None

#         self.validation_token_ids = None

#     # ======================================================
#     # Tokenize text
#     # ======================================================

#     def _tokenize(self, text):
#         """
#         Convert text into a LongTensor of token IDs.
#         """

#         token_ids = self.tokenizer.encode(
#             text,
#             add_bos=True,
#             add_eos=True
#         )

#         if len(token_ids) <= self.max_seq_len:

#             raise ValueError(
#                 "Tokenized text is too short for "
#                 f"max_seq_len={self.max_seq_len}"
#             )

#         return torch.tensor(
#             token_ids,
#             dtype=torch.long
#         )

#     # ======================================================
#     # Build datasets
#     # ======================================================

#     # def build_datasets(self):

#     #     # --------------------------------------------------
#     #     # Split corpus
#     #     # --------------------------------------------------

#     #     train_text, validation_text = (
#     #         self.corpus.train_validation_split()
#     #     )

#     #     # --------------------------------------------------
#     #     # Tokenize training text
#     #     # --------------------------------------------------

#     #     self.train_token_ids = self._tokenize(
#     #         train_text
#     #     )

#     #     # --------------------------------------------------
#     #     # Tokenize validation text
#     #     # --------------------------------------------------

#     #     self.validation_token_ids = self._tokenize(
#     #         validation_text
#     #     )

#     #     # --------------------------------------------------
#     #     # Create training dataset
#     #     # --------------------------------------------------

#     #     self.train_dataset = GenieeTextDataset(
#     #         token_ids=self.train_token_ids,
#     #         max_seq_len=self.max_seq_len,
#     #         stride=self.stride
#     #     )

#     #     # --------------------------------------------------
#     #     # Create validation dataset
#     #     # --------------------------------------------------

#     #     self.validation_dataset = GenieeTextDataset(
#     #         token_ids=self.validation_token_ids,
#     #         max_seq_len=self.max_seq_len,
#     #         stride=self.stride
#     #     )

#     #     return (
#     #         self.train_dataset,
#     #         self.validation_dataset
#     #     )
#     def build_datasets(self):

#         # --------------------------------------------------
#         # Split corpus into documents
#         # --------------------------------------------------

#         documents = self.corpus.load_documents()

#         # --------------------------------------------------
#         # Deterministic document-level split
#         # --------------------------------------------------

#         import random

#         documents = list(documents)

#         generator = random.Random(42)

#         generator.shuffle(documents)

#         validation_count = max(
#             1,
#             int(len(documents) * self.validation_split)
#         )

#         if validation_count >= len(documents):
#             validation_count = len(documents) - 1

#         validation_documents = documents[
#             :validation_count
#         ]

#         train_documents = documents[
#             validation_count:
#         ]

#         # --------------------------------------------------
#         # Tokenize each document separately
#         # --------------------------------------------------

#         def tokenize_documents(documents):

#             all_token_ids = []

#             for document in documents:

#                 token_ids = self.tokenizer.encode(
#                     document["text"],
#                     add_bos=True,
#                     add_eos=True
#                 )

#                 all_token_ids.extend(
#                     token_ids
#                 )

#             if len(all_token_ids) <= self.max_seq_len:

#                 raise ValueError(
#                     "Tokenized documents are too short for "
#                     f"max_seq_len={self.max_seq_len}"
#                 )

#             return torch.tensor(
#                 all_token_ids,
#                 dtype=torch.long
#             )

#         # --------------------------------------------------
#         # Tokenize training documents
#         # --------------------------------------------------

#         self.train_token_ids = tokenize_documents(
#             train_documents
#         )

#         # --------------------------------------------------
#         # Tokenize validation documents
#         # --------------------------------------------------

#         self.validation_token_ids = tokenize_documents(
#             validation_documents
#         )

#         # --------------------------------------------------
#         # Create training dataset
#         # --------------------------------------------------

#         self.train_dataset = GenieeTextDataset(
#             token_ids=self.train_token_ids,
#             max_seq_len=self.max_seq_len,
#             stride=self.stride
#         )

#         # --------------------------------------------------
#         # Create validation dataset
#         # --------------------------------------------------

#         self.validation_dataset = GenieeTextDataset(
#             token_ids=self.validation_token_ids,
#             max_seq_len=self.max_seq_len,
#             stride=self.stride
#         )

#         return (
#             self.train_dataset,
#             self.validation_dataset
#         )

#     # ======================================================
#     # Build DataLoaders
#     # ======================================================

#     def build_dataloaders(self):

#         # --------------------------------------------------
#         # Build datasets first
#         # --------------------------------------------------

#         if self.train_dataset is None:

#             self.build_datasets()

#         # --------------------------------------------------
#         # Training DataLoader
#         # --------------------------------------------------

#         self.train_loader = DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=self.shuffle_train,
#             drop_last=False
#         )

#         # --------------------------------------------------
#         # Validation DataLoader
#         # --------------------------------------------------

#         self.validation_loader = DataLoader(
#             self.validation_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             drop_last=False
#         )

#         return (
#             self.train_loader,
#             self.validation_loader
#         )

#     # ======================================================
#     # Build everything
#     # ======================================================

#     def build(self):

#         return self.build_dataloaders()

#     # ======================================================
#     # Information
#     # ======================================================

#     def summary(self):

#         if self.train_dataset is None:

#             self.build()

#         # print()
#         # print("=" * 60)
#         # print("GENIEE DATA MODULE")
#         # print("=" * 60)

#         # print(
#         #     f"Tokenizer vocabulary : "
#         #     f"{self.tokenizer.vocab_size}"
#         # )

#         # print(
#         #     f"Max sequence length : "
#         #     f"{self.max_seq_len}"
#         # )

#         # print(
#         #     f"Batch size           : "
#         #     f"{self.batch_size}"
#         # )

#         # print(
#         #     f"Stride               : "
#         #     f"{self.stride}"
#         # )

#         # print(
#         #     f"Training tokens      : "
#         #     f"{len(self.train_token_ids)}"
#         # )

#         # print(
#         #     f"Validation tokens    : "
#         #     f"{len(self.validation_token_ids)}"
#         # )

#         # print(
#         #     f"Training samples     : "
#         #     f"{len(self.train_dataset)}"
#         # )

#         # print(
#         #     f"Validation samples   : "
#         #     f"{len(self.validation_dataset)}"
#         # )

#         # print("=" * 60)
# from pathlib import Path

# import torch

# from tokenizer.tokenizer import GenieeTokenizer
# from data.corpus import GenieeCorpus
# from data.dataset import GenieeTextDataset


# class GenieeDataModule:

#     def __init__(
#         self,
#         corpus_dir="corpus",
#         tokenizer_path="tokenizer/artifacts/geniee.model",
#         max_seq_len=128,
#         stride=64,
#         validation_split=0.1
#     ):

#         self.corpus_dir = Path(
#             corpus_dir
#         )

#         self.tokenizer_path = Path(
#             tokenizer_path
#         )

#         self.max_seq_len = max_seq_len
#         self.stride = stride
#         self.validation_split = validation_split


#         # ------------------------------------------------------
#         # Tokenizer
#         # ------------------------------------------------------

#         self.tokenizer = GenieeTokenizer(
#             self.tokenizer_path
#         )


#         # ------------------------------------------------------
#         # Corpus
#         # ------------------------------------------------------

#         self.corpus = GenieeCorpus(
#             corpus_dir=self.corpus_dir,
#             validation_split=self.validation_split,
#             seed=42
#         )


#         self.train_token_ids = None
#         self.validation_token_ids = None

#         self.train_dataset = None
#         self.validation_dataset = None


#     # ==========================================================
#     # Tokenize individual documents
#     # ==========================================================

#     def _tokenize_documents(
#         self,
#         documents
#     ):

#         all_token_ids = []


#         for document in documents:

#             token_ids = self.tokenizer.encode(
#                 document["text"],
#                 add_bos=True,
#                 add_eos=True
#             )


#             # Preserve document boundaries by placing
#             # each document between BOS and EOS.

#             all_token_ids.extend(
#                 token_ids
#             )


#         if len(all_token_ids) <= self.max_seq_len:

#             raise ValueError(
#                 "Tokenized documents are too short "
#                 f"for max_seq_len={self.max_seq_len}"
#             )


#         return torch.tensor(
#             all_token_ids,
#             dtype=torch.long
#         )


#     # ==========================================================
#     # Build datasets
#     # ==========================================================

#     def build_datasets(self):

#         (
#             train_documents,
#             validation_documents
#         ) = self.corpus.train_validation_documents()


#         # ------------------------------------------------------
#         # Tokenize train documents
#         # ------------------------------------------------------

#         self.train_token_ids = (
#             self._tokenize_documents(
#                 train_documents
#             )
#         )


#         # ------------------------------------------------------
#         # Tokenize validation documents
#         # ------------------------------------------------------

#         self.validation_token_ids = (
#             self._tokenize_documents(
#                 validation_documents
#             )
#         )


#         # ------------------------------------------------------
#         # Create training dataset
#         # ------------------------------------------------------

#         self.train_dataset = GenieeTextDataset(
#             token_ids=self.train_token_ids,
#             max_seq_len=self.max_seq_len,
#             stride=self.stride
#         )


#         # ------------------------------------------------------
#         # Create validation dataset
#         # ------------------------------------------------------

#         self.validation_dataset = GenieeTextDataset(
#             token_ids=self.validation_token_ids,
#             max_seq_len=self.max_seq_len,
#             stride=self.stride
#         )


#         return (
#             self.train_dataset,
#             self.validation_dataset
#         )

# from pathlib import Path
# import sys
# import re

# import torch
# from torch.utils.data import DataLoader


# # ==========================================================
# # Project root
# # ==========================================================

# PROJECT_ROOT = (
#     Path(__file__).resolve().parent.parent
# )

# if str(PROJECT_ROOT) not in sys.path:

#     sys.path.insert(
#         0,
#         str(PROJECT_ROOT)
#     )


# # ==========================================================
# # Geniee imports
# # ==========================================================

# from data.corpus import GenieeCorpus

# from data.dataset import (
#     GenieeTextDataset
# )

# from tokenizer.tokenizer import (
#     GenieeTokenizer
# )


# # ==========================================================
# # Data Module
# # ==========================================================

# class GenieeDataModule:

#     def __init__(
#         self,
#         corpus_dir,
#         tokenizer_path,
#         max_seq_len,
#         batch_size,
#         stride=None,
#         validation_split=0.1,
#         shuffle_train=True
#     ):

#         self.corpus_dir = Path(
#             corpus_dir
#         )

#         self.tokenizer_path = Path(
#             tokenizer_path
#         )

#         self.max_seq_len = (
#             max_seq_len
#         )

#         self.batch_size = (
#             batch_size
#         )

#         self.stride = stride

#         self.validation_split = (
#             validation_split
#         )

#         self.shuffle_train = (
#             shuffle_train
#         )

#         # --------------------------------------------------
#         # Tokenizer
#         # --------------------------------------------------

#         self.tokenizer = GenieeTokenizer(
#             model_path=self.tokenizer_path
#         )

#         # --------------------------------------------------
#         # Corpus
#         # --------------------------------------------------

#         self.corpus = GenieeCorpus(
#             corpus_dir=self.corpus_dir
#         )

#         self.train_dataset = None

#         self.validation_dataset = None

#         self.train_loader = None

#         self.validation_loader = None

#         self.train_examples = []

#         self.validation_examples = []

#     # ======================================================
#     # Extract independent examples
#     # ======================================================

#     @staticmethod
#     def _extract_examples(
#         text
#     ):
#         """
#         Extract independent User/Assistant examples.

#         Example:

#         User: What is Selenium?
#         Assistant: Selenium is...

#         User: What is Playwright?
#         Assistant: Playwright is...

#         becomes:

#         Example 1:
#         User: What is Selenium?
#         Assistant: Selenium is...

#         Example 2:
#         User: What is Playwright?
#         Assistant: Playwright is...
#         """

#         text = text.strip()

#         if not text:

#             return []

#         # --------------------------------------------------
#         # Look for User:/Assistant: structure
#         # --------------------------------------------------

#         pattern = re.compile(
#             r"(?ms)"
#             r"^User:\s*(.*?)"
#             r"^\s*Assistant:\s*(.*?)"
#             r"(?=^\s*User:|\Z)"
#         )

#         matches = pattern.findall(
#             text
#         )

#         examples = []

#         for question, answer in matches:

#             question = question.strip()

#             answer = answer.strip()

#             if not question or not answer:
#                 continue

#             example = (
#                 "User: "
#                 + question
#                 + "\n"
#                 + "Assistant: "
#                 + answer
#             )

#             examples.append(
#                 example
#             )

#         # --------------------------------------------------
#         # If this is not a Q&A document,
#         # treat the document itself as one example.
#         # --------------------------------------------------

#         if not examples:

#             examples.append(
#                 text
#             )

#         return examples

#     # ======================================================
#     # Convert documents to independent examples
#     # ======================================================

#     def build_datasets(self):

#         print()
#         print("Loading corpus...")

#         # (train_documents,validation_documents) = self.corpus.train_validation_documents(validation_split=self.validation_split)
#         (train_documents, validation_documents) = (self.corpus.train_validation_documents())

#         print(f"Train documents : "f"{len(train_documents)}")

#         print(
#             f"Validation documents : "
#             f"{len(validation_documents)}"
#         )

#         # --------------------------------------------------
#         # Convert documents into independent examples
#         # --------------------------------------------------

#         self.train_examples = (
#             self._documents_to_examples(
#                 train_documents
#             )
#         )

#         self.validation_examples = (
#             self._documents_to_examples(
#                 validation_documents
#             )
#         )

#         print(
#             f"Train examples       : "
#             f"{len(self.train_examples)}"
#         )

#         print(
#             f"Validation examples  : "
#             f"{len(self.validation_examples)}"
#         )

#         # --------------------------------------------------
#         # Tokenize
#         # --------------------------------------------------

#         self.train_token_ids = (
#             self._tokenize_examples(
#                 self.train_examples
#             )
#         )

#         self.validation_token_ids = (
#             self._tokenize_examples(
#                 self.validation_examples
#             )
#         )

#         print(
#             f"Train token sequences      : "
#             f"{len(self.train_token_ids)}"
#         )

#         print(
#             f"Validation token sequences : "
#             f"{len(self.validation_token_ids)}"
#         )

#         # --------------------------------------------------
#         # Create datasets
#         # --------------------------------------------------

#         self.train_dataset = GenieeTextDataset(
#             sequences=self.train_token_ids,
#             max_seq_len=self.max_seq_len,
#             pad_id=self.tokenizer.pad_id
#         )

#         self.validation_dataset = GenieeTextDataset(
#             sequences=self.validation_token_ids,
#             max_seq_len=self.max_seq_len,
#             pad_id=self.tokenizer.pad_id
#         )

#         return (
#             self.train_dataset,
#             self.validation_dataset
#         )
    
#     def _documents_to_examples(
#         self,
#         documents
#     ):

#         examples = []

#         for document in documents:

#             text = document[
#                 "text"
#             ]

#             document_examples = (
#                 self._extract_examples(
#                     text
#                 )
#             )

#             for example in document_examples:

#                 examples.append(
#                     example
#                 )

#         return examples

#     # ======================================================
#     # Tokenize examples
#     # ======================================================

#     def _tokenize_examples(
#         self,
#         examples
#     ):

#         tokenized = []

#         for example in examples:

#             token_ids = self.tokenizer.encode(

#                 example,

#                 add_bos=True,

#                 add_eos=True
#             )

#             if len(token_ids) < 2:
#                 continue

#             tokenized.append(
#                 token_ids
#             )

#         return tokenized

#     # ======================================================
#     # Build datasets
#     # ======================================================

#     # def build(self):

#     #     print()

#     #     print(
#     #         "Loading corpus..."
#     #     )

#     #     (train_documents,validation_documents) = self.corpus.train_validation_documents(validation_split=self.validation_split)

#     #     print(f"Train documents  : "f"{len(train_documents)}")

#     #     print(
#     #         f"Validation documents : "
#     #         f"{len(validation_documents)}"
#     #     )

#     #     # --------------------------------------------------
#     #     # Convert documents into independent examples
#     #     # --------------------------------------------------

#     #     self.train_examples = (
#     #         self._documents_to_examples(
#     #             train_documents
#     #         )
#     #     )

#     #     self.validation_examples = (
#     #         self._documents_to_examples(
#     #             validation_documents
#     #         )
#     #     )

#     #     print(
#     #         f"Train examples       : "
#     #         f"{len(self.train_examples)}"
#     #     )

#     #     print(
#     #         f"Validation examples  : "
#     #         f"{len(self.validation_examples)}"
#     #     )

#     #     # --------------------------------------------------
#     #     # Tokenization
#     #     # --------------------------------------------------

#     #     train_sequences = (
#     #         self._tokenize_examples(
#     #             self.train_examples
#     #         )
#     #     )

#     #     validation_sequences = (
#     #         self._tokenize_examples(
#     #             self.validation_examples
#     #         )
#     #     )

#     #     print(
#     #         f"Train token sequences      : "
#     #         f"{len(train_sequences)}"
#     #     )

#     #     print(
#     #         f"Validation token sequences : "
#     #         f"{len(validation_sequences)}"
#     #     )

#     #     if not train_sequences:

#     #         raise ValueError(
#     #             "No training sequences were created."
#     #         )

#     #     if not validation_sequences:

#     #         raise ValueError(
#     #             "No validation sequences were created."
#     #         )

#     #     # --------------------------------------------------
#     #     # Create datasets
#     #     # --------------------------------------------------

#     #     self.train_dataset = (
#     #         GenieeTextDataset(
#     #             sequences=train_sequences,
#     #             max_seq_len=self.max_seq_len,
#     #             pad_id=self.tokenizer.pad_id
#     #         )
#     #     )

#     #     self.validation_dataset = (
#     #         GenieeTextDataset(
#     #             sequences=validation_sequences,
#     #             max_seq_len=self.max_seq_len,
#     #             pad_id=self.tokenizer.pad_id
#     #         )
#     #     )

#     #     # --------------------------------------------------
#     #     # Create loaders
#     #     # --------------------------------------------------

#     #     self.train_loader = DataLoader(

#     #         self.train_dataset,

#     #         batch_size=self.batch_size,

#     #         shuffle=self.shuffle_train
#     #     )

#     #     self.validation_loader = DataLoader(

#     #         self.validation_dataset,

#     #         batch_size=self.batch_size,

#     #         shuffle=False
#     #     )

#     #     return (
#     #         self.train_loader,
#     #         self.validation_loader
#     #     )
#     def build(self):

#         (
#             self.train_dataset,
#             self.validation_dataset
#         ) = self.build_datasets()

#         self.train_loader = DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=self.shuffle_train
#         )

#         self.validation_loader = DataLoader(
#             self.validation_dataset,
#             batch_size=self.batch_size,
#             shuffle=False
#         )

#         return (
#             self.train_loader,
#             self.validation_loader
#         )

#     # ======================================================
#     # Summary
#     # ======================================================

#     def summary(self):

#         print()

#         print(
#             "=" * 60
#         )

#         print(
#             "GENIEE DATASET SUMMARY"
#         )

#         print(
#             "=" * 60
#         )

#         if self.train_dataset is not None:

#             print(
#                 f"Train examples       : "
#                 f"{len(self.train_dataset)}"
#             )

#         if self.validation_dataset is not None:

#             print(
#                 f"Validation examples  : "
#                 f"{len(self.validation_dataset)}"
#             )

#         print(
#             f"Max sequence length  : "
#             f"{self.max_seq_len}"
#         )

#         print(
#             f"Batch size            : "
#             f"{self.batch_size}"
#         )

#         print(
#             f"Vocabulary size       : "
#             f"{self.tokenizer.vocab_size}"
#         )

#         print(
#             f"PAD token ID          : "
#             f"{self.tokenizer.pad_id}"
#         )

#         print(
#             "=" * 60
#         )


from pathlib import Path
import sys
import re

from torch.utils.data import DataLoader


# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ==========================================================
# Geniee imports
# ==========================================================

from data.corpus import GenieeCorpus

from data.dataset import (
    GenieeTextDataset
)

from tokenizer.tokenizer import (
    GenieeTokenizer
)


# ==========================================================
# Data Module
# ==========================================================

class GenieeDataModule:

    def __init__(
        self,
        corpus_dir,
        tokenizer_path,
        max_seq_len,
        batch_size,
        stride=None,
        validation_split=0.1,
        shuffle_train=True,
        seed=42
    ):

        self.corpus_dir = Path(
            corpus_dir
        )

        self.tokenizer_path = Path(
            tokenizer_path
        )

        self.max_seq_len = (
            max_seq_len
        )

        self.batch_size = (
            batch_size
        )

        self.stride = stride

        self.validation_split = (
            validation_split
        )

        self.shuffle_train = (
            shuffle_train
        )

        self.seed = seed

        # --------------------------------------------------
        # Tokenizer
        # --------------------------------------------------

        self.tokenizer = GenieeTokenizer(
            model_path=self.tokenizer_path
        )

        # --------------------------------------------------
        # Corpus
        #
        # IMPORTANT:
        # Use the SAME validation split and seed that
        # were requested by the DataModule.
        # --------------------------------------------------

        self.corpus = GenieeCorpus(
            corpus_dir=self.corpus_dir,
            validation_split=self.validation_split,
            seed=self.seed
        )

        # --------------------------------------------------
        # Runtime objects
        # --------------------------------------------------

        self.train_dataset = None

        self.validation_dataset = None

        self.train_loader = None

        self.validation_loader = None

        self.train_examples = []

        self.validation_examples = []

        self.train_token_ids = []

        self.validation_token_ids = []

        self.train_documents = []

        self.validation_documents = []

    # ======================================================
    # Extract independent User/Assistant examples
    # ======================================================

    @staticmethod
    def _extract_examples(text):
        """
        Extract independent User/Assistant examples.

        Example:

        User: What is Selenium?
        Assistant: Selenium is...

        User: What is Playwright?
        Assistant: Playwright is...

        becomes two independent examples.
        """

        text = text.strip()

        if not text:

            return []

        # --------------------------------------------------
        # User / Assistant pattern
        # --------------------------------------------------

        pattern = re.compile(
            r"(?ms)"
            r"^User:\s*(.*?)"
            r"^\s*Assistant:\s*(.*?)"
            r"(?=^\s*User:|\Z)"
        )

        matches = pattern.findall(
            text
        )

        examples = []

        for question, answer in matches:

            question = question.strip()

            answer = answer.strip()

            if not question or not answer:

                continue

            example = (
                "User: "
                + question
                + "\n"
                + "Assistant: "
                + answer
            )

            examples.append(
                example
            )

        # --------------------------------------------------
        # Non-Q&A document
        # --------------------------------------------------

        if not examples:

            examples.append(
                text
            )

        return examples

    # ======================================================
    # Convert documents to independent examples
    # ======================================================

    def _documents_to_examples(
        self,
        documents
    ):

        examples = []

        for document in documents:

            text = document["text"]

            document_examples = (
                self._extract_examples(
                    text
                )
            )

            examples.extend(
                document_examples
            )

        return examples

    # ======================================================
    # Tokenize examples
    # ======================================================

    def _tokenize_examples(
        self,
        examples
    ):

        tokenized = []

        for example in examples:

            token_ids = self.tokenizer.encode(
                example,
                add_bos=True,
                add_eos=True
            )

            if len(token_ids) < 2:

                continue

            tokenized.append(
                token_ids
            )

        return tokenized

    # ======================================================
    # Build datasets
    # ======================================================

    def build_datasets(self):

        print()
        print("Loading corpus...")

        # --------------------------------------------------
        # IMPORTANT:
        # This now uses the validation_split and seed that
        # were passed to GenieeDataModule.
        # --------------------------------------------------

        (
            train_documents,
            validation_documents
        ) = self.corpus.train_validation_documents()

        # Save document split for later inspection/debugging.

        self.train_documents = (
            train_documents
        )

        self.validation_documents = (
            validation_documents
        )

        print(
            f"Train documents : "
            f"{len(train_documents)}"
        )

        print(
            f"Validation documents : "
            f"{len(validation_documents)}"
        )

        # --------------------------------------------------
        # Convert documents into independent examples
        # --------------------------------------------------

        self.train_examples = (
            self._documents_to_examples(
                train_documents
            )
        )

        self.validation_examples = (
            self._documents_to_examples(
                validation_documents
            )
        )

        print(
            f"Train examples       : "
            f"{len(self.train_examples)}"
        )

        print(
            f"Validation examples  : "
            f"{len(self.validation_examples)}"
        )

        # --------------------------------------------------
        # Tokenize
        # --------------------------------------------------

        self.train_token_ids = (
            self._tokenize_examples(
                self.train_examples
            )
        )

        self.validation_token_ids = (
            self._tokenize_examples(
                self.validation_examples
            )
        )

        print(
            f"Train token sequences      : "
            f"{len(self.train_token_ids)}"
        )

        print(
            f"Validation token sequences : "
            f"{len(self.validation_token_ids)}"
        )

        # --------------------------------------------------
        # Safety checks
        # --------------------------------------------------

        if not self.train_token_ids:

            raise ValueError(
                "No training sequences were created."
            )

        if not self.validation_token_ids:

            raise ValueError(
                "No validation sequences were created."
            )

        # --------------------------------------------------
        # Create independent datasets
        # --------------------------------------------------

        self.train_dataset = GenieeTextDataset(
            sequences=self.train_token_ids,
            max_seq_len=self.max_seq_len,
            pad_id=self.tokenizer.pad_id
        )

        self.validation_dataset = GenieeTextDataset(
            sequences=self.validation_token_ids,
            max_seq_len=self.max_seq_len,
            pad_id=self.tokenizer.pad_id
        )

        return (
            self.train_dataset,
            self.validation_dataset
        )

    # ======================================================
    # Build DataLoaders
    # ======================================================

    def build(self):

        (
            self.train_dataset,
            self.validation_dataset
        ) = self.build_datasets()

        # --------------------------------------------------
        # Training loader
        # --------------------------------------------------

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train
        )

        # --------------------------------------------------
        # Validation loader
        # --------------------------------------------------

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return (
            self.train_loader,
            self.validation_loader
        )

    # ======================================================
    # Summary
    # ======================================================

    def summary(self):

        print()
        print("=" * 60)
        print("GENIEE DATASET SUMMARY")
        print("=" * 60)

        if self.train_dataset is not None:

            print(
                f"Train dataset sequences : "
                f"{len(self.train_dataset)}"
            )

        if self.validation_dataset is not None:

            print(
                f"Validation dataset sequences : "
                f"{len(self.validation_dataset)}"
            )

        print(
            f"Train examples : "
            f"{len(self.train_examples)}"
        )

        print(
            f"Validation examples : "
            f"{len(self.validation_examples)}"
        )

        print(
            f"Max sequence length : "
            f"{self.max_seq_len}"
        )

        print(
            f"Batch size : "
            f"{self.batch_size}"
        )

        print(
            f"Vocabulary size : "
            f"{self.tokenizer.vocab_size}"
        )

        print(
            f"PAD token ID : "
            f"{self.tokenizer.pad_id}"
        )

        print(
            f"Validation split : "
            f"{self.validation_split}"
        )

        print(
            f"Random seed : "
            f"{self.seed}"
        )

        print("=" * 60)