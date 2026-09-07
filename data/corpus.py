# # from pathlib import Path


# # class GenieeCorpus:
# #     """
# #     Loads and manages the text corpus used by Geniee.

# #     Responsibilities:
# #         1. Locate corpus files
# #         2. Read text
# #         3. Clean basic whitespace
# #         4. Combine multiple text files
# #         5. Create train/validation splits
# #     """

# #     def __init__(
# #         self,
# #         corpus_dir,
# #         validation_split=0.1
# #     ):
# #         self.corpus_dir = Path(corpus_dir)

# #         if not self.corpus_dir.exists():
# #             raise FileNotFoundError(
# #                 f"Corpus directory not found: "
# #                 f"{self.corpus_dir}"
# #             )

# #         if not self.corpus_dir.is_dir():
# #             raise ValueError(
# #                 f"Corpus path is not a directory: "
# #                 f"{self.corpus_dir}"
# #             )

# #         if not 0 < validation_split < 1:
# #             raise ValueError(
# #                 "validation_split must be between 0 and 1"
# #             )

# #         self.validation_split = validation_split

# #     # --------------------------------------------------
# #     # Find text files
# #     # --------------------------------------------------

# #     def get_text_files(self):
# #         """
# #         Return all .txt files in the corpus directory.
# #         """

# #         files = sorted(
# #             self.corpus_dir.glob("*.txt")
# #         )

# #         if not files:
# #             raise FileNotFoundError(
# #                 f"No .txt files found in "
# #                 f"{self.corpus_dir}"
# #             )

# #         return files

# #     # --------------------------------------------------
# #     # Read one file
# #     # --------------------------------------------------

# #     @staticmethod
# #     def read_file(file_path):
# #         """
# #         Read a single text file.
# #         """

# #         file_path = Path(file_path)

# #         if not file_path.exists():
# #             raise FileNotFoundError(
# #                 f"Corpus file not found: {file_path}"
# #             )

# #         text = file_path.read_text(
# #             encoding="utf-8"
# #         )

# #         return text

# #     # --------------------------------------------------
# #     # Basic cleaning
# #     # --------------------------------------------------

# #     @staticmethod
# #     def clean_text(text):
# #         """
# #         Perform basic text normalization.

# #         We intentionally keep this conservative.
# #         Aggressive cleaning can remove useful
# #         language-model training information.
# #         """

# #         lines = []

# #         for line in text.splitlines():

# #             line = line.strip()

# #             if not line:
# #                 continue

# #             lines.append(line)

# #         return "\n".join(lines)

# #     # --------------------------------------------------
# #     # Load all documents
# #     # --------------------------------------------------

# #     def load_documents(self):
# #         """
# #         Load and clean all corpus documents.

# #         Returns:
# #             list[str]
# #         """

# #         documents = []

# #         for file_path in self.get_text_files():

# #             text = self.read_file(
# #                 file_path
# #             )

# #             text = self.clean_text(
# #                 text
# #             )

# #             if text:
# #                 documents.append(text)

# #         if not documents:
# #             raise ValueError(
# #                 "Corpus contains no usable text."
# #             )

# #         return documents

# #     # --------------------------------------------------
# #     # Combine corpus
# #     # --------------------------------------------------

# #     def load_text(self):
# #         """
# #         Combine all corpus documents into one text.
# #         """

# #         documents = self.load_documents()

# #         return "\n\n".join(
# #             documents
# #         )

# #     # --------------------------------------------------
# #     # Train / validation split
# #     # --------------------------------------------------

# #     def train_validation_split(self):
# #         """
# #         Split corpus into train and validation text.

# #         The split is performed at the character level
# #         for this initial implementation.
# #         """

# #         text = self.load_text()

# #         split_index = int(
# #             len(text)
# #             * (1 - self.validation_split)
# #         )

# #         train_text = text[:split_index]

# #         validation_text = text[split_index:]

# #         if not train_text:
# #             raise ValueError(
# #                 "Training corpus is empty."
# #             )

# #         if not validation_text:
# #             raise ValueError(
# #                 "Validation corpus is empty."
# #             )

# #         return (
# #             train_text,
# #             validation_text
# #         )

# from pathlib import Path
# import random


# class GenieeCorpus:
#     """
#     Loads and manages the text corpus used by Geniee.

#     Responsibilities:
#         1. Recursively locate corpus files
#         2. Read text files
#         3. Clean basic whitespace
#         4. Combine multiple documents
#         5. Create train/validation splits
#         6. Keep document boundaries intact
#     """

#     def __init__(
#         self,
#         corpus_dir,
#         validation_split=0.1,
#         seed=42
#     ):
#         self.corpus_dir = Path(corpus_dir)

#         if not self.corpus_dir.exists():
#             raise FileNotFoundError(
#                 f"Corpus directory not found: "
#                 f"{self.corpus_dir}"
#             )

#         if not self.corpus_dir.is_dir():
#             raise ValueError(
#                 f"Corpus path is not a directory: "
#                 f"{self.corpus_dir}"
#             )

#         if not 0 < validation_split < 1:
#             raise ValueError(
#                 "validation_split must be between 0 and 1"
#             )

#         self.validation_split = validation_split
#         self.seed = seed

#     # --------------------------------------------------
#     # Find text files
#     # --------------------------------------------------

#     def get_text_files(self):
#         """
#         Recursively find all .txt files.

#         Example:

#             corpus/
#                 ai/
#                     transformers.txt
#                 programming/
#                     python.txt
#                 general/
#                     science.txt

#         All .txt files will be discovered.
#         """

#         files = sorted(
#             self.corpus_dir.rglob("*.txt")
#         )

#         if not files:
#             raise FileNotFoundError(
#                 f"No .txt files found in "
#                 f"{self.corpus_dir}"
#             )

#         return files

#     # --------------------------------------------------
#     # Read one file
#     # --------------------------------------------------

#     @staticmethod
#     def read_file(file_path):
#         """
#         Read a single UTF-8 text file.
#         """

#         file_path = Path(file_path)

#         if not file_path.exists():
#             raise FileNotFoundError(
#                 f"Corpus file not found: "
#                 f"{file_path}"
#             )

#         if not file_path.is_file():
#             raise ValueError(
#                 f"Corpus path is not a file: "
#                 f"{file_path}"
#             )

#         text = file_path.read_text(
#             encoding="utf-8"
#         )

#         return text

#     # --------------------------------------------------
#     # Basic cleaning
#     # --------------------------------------------------

#     @staticmethod
#     def clean_text(text):
#         """
#         Perform conservative text normalization.

#         We preserve punctuation and sentence structure.
#         """

#         if not isinstance(text, str):
#             raise TypeError(
#                 "text must be a string"
#             )

#         lines = []

#         for line in text.splitlines():

#             line = line.strip()

#             if not line:
#                 continue

#             lines.append(line)

#         return "\n".join(lines)

#     # --------------------------------------------------
#     # Load documents
#     # --------------------------------------------------

#     def load_documents(self):
#         """
#         Load and clean all corpus documents.

#         Returns:
#             list[dict]

#         Each document contains:

#             {
#                 "path": Path(...),
#                 "text": "..."
#             }
#         """

#         documents = []

#         for file_path in self.get_text_files():

#             text = self.read_file(
#                 file_path
#             )

#             text = self.clean_text(
#                 text
#             )

#             if not text:
#                 continue

#             documents.append(
#                 {
#                     "path": file_path,
#                     "text": text
#                 }
#             )

#         if not documents:
#             raise ValueError(
#                 "Corpus contains no usable text."
#             )

#         return documents

#     # --------------------------------------------------
#     # Combine documents
#     # --------------------------------------------------

#     @staticmethod
#     def combine_documents(documents):
#         """
#         Combine document texts while preserving
#         clear boundaries between documents.
#         """

#         return "\n\n".join(
#             document["text"]
#             for document in documents
#         )

#     # --------------------------------------------------
#     # Load complete corpus
#     # --------------------------------------------------

#     def load_text(self):
#         """
#         Load and combine the complete corpus.
#         """

#         documents = self.load_documents()

#         return self.combine_documents(
#             documents
#         )

#     # --------------------------------------------------
#     # Train / validation split
#     # --------------------------------------------------

#     def train_validation_split(self):
#         """
#         Split documents into training and validation sets.

#         The split happens at document level rather than
#         character level.

#         This prevents validation data from being created
#         by cutting through the middle of a document.
#         """

#         documents = self.load_documents()

#         if len(documents) < 2:

#             raise ValueError(
#                 "At least 2 corpus documents are required "
#                 "for a train/validation split."
#             )

#         documents = list(documents)

#         random_generator = random.Random(
#             self.seed
#         )

#         random_generator.shuffle(
#             documents
#         )

#         validation_count = max(
#             1,
#             int(
#                 len(documents)
#                 * self.validation_split
#             )
#         )

#         if validation_count >= len(documents):

#             validation_count = (
#                 len(documents) - 1
#             )

#         validation_documents = (
#             documents[
#                 :validation_count
#             ]
#         )

#         train_documents = (
#             documents[
#                 validation_count:
#             ]
#         )

#         train_text = self.combine_documents(
#             train_documents
#         )

#         validation_text = self.combine_documents(
#             validation_documents
#         )

#         if not train_text:
#             raise ValueError(
#                 "Training corpus is empty."
#             )

#         if not validation_text:
#             raise ValueError(
#                 "Validation corpus is empty."
#             )

#         return (
#             train_text,
#             validation_text
#         )

#     # --------------------------------------------------
#     # Corpus statistics
#     # --------------------------------------------------

#     def statistics(self):
#         """
#         Return basic corpus statistics.
#         """

#         documents = self.load_documents()

#         total_characters = sum(
#             len(document["text"])
#             for document in documents
#         )

#         total_words = sum(
#             len(
#                 document["text"].split()
#             )
#             for document in documents
#         )

#         return {
#             "files": len(documents),
#             "characters": total_characters,
#             "words": total_words
#         }

# from pathlib import Path
# import random


# class GenieeCorpus:

#     def __init__(
#         self,
#         corpus_dir="corpus",
#         validation_split=0.1,
#         seed=42
#     ):

#         self.corpus_dir = Path(corpus_dir)

#         self.validation_split = validation_split
#         self.seed = seed

#         if not self.corpus_dir.exists():

#             raise FileNotFoundError(
#                 f"Corpus directory not found: {self.corpus_dir}"
#             )


#     # ==========================================================
#     # Load all text documents
#     # ==========================================================

#     def load_documents(self):

#         documents = []

#         for path in sorted(
#             self.corpus_dir.rglob("*.txt")
#         ):

#             try:

#                 text = path.read_text(
#                     encoding="utf-8"
#                 )

#             except UnicodeDecodeError:

#                 print(
#                     f"Skipping non UTF-8 file: {path}"
#                 )

#                 continue


#             # Remove empty lines while preserving
#             # the actual document content.

#             lines = [
#                 line.strip()
#                 for line in text.splitlines()
#                 if line.strip()
#             ]

#             text = "\n".join(lines)

#             if not text:

#                 continue


#             # Identify the document category.

#             relative_path = path.relative_to(
#                 self.corpus_dir
#             )

#             category = (
#                 relative_path.parts[0]
#                 if len(relative_path.parts) > 1
#                 else "root"
#             )


#             documents.append(
#                 {
#                     "path": path,
#                     "category": category,
#                     "text": text
#                 }
#             )


#         if not documents:

#             raise ValueError(
#                 f"No .txt documents found in "
#                 f"{self.corpus_dir}"
#             )


#         return documents


#     # ==========================================================
#     # Train / validation document split
#     # ==========================================================

#     def train_validation_documents(self):

#         documents = self.load_documents()

#         documents = list(documents)

#         generator = random.Random(
#             self.seed
#         )

#         generator.shuffle(
#             documents
#         )


#         validation_count = max(
#             1,
#             int(
#                 len(documents)
#                 * self.validation_split
#             )
#         )


#         if validation_count >= len(documents):

#             validation_count = (
#                 len(documents) - 1
#             )


#         validation_documents = documents[
#             :validation_count
#         ]

#         train_documents = documents[
#             validation_count:
#         ]


#         return (
#             train_documents,
#             validation_documents
#         )


#     # ==========================================================
#     # Backward-compatible text split
#     # ==========================================================

#     def train_validation_split(self):

#         (
#             train_documents,
#             validation_documents
#         ) = self.train_validation_documents()


#         train_text = "\n".join(
#             document["text"]
#             for document in train_documents
#         )


#         validation_text = "\n".join(
#             document["text"]
#             for document in validation_documents
#         )


#         return (
#             train_text,
#             validation_text
#         )



from pathlib import Path
import random
import re


class GenieeCorpus:

    def __init__(
        self,
        corpus_dir="corpus",
        validation_split=0.1,
        seed=42
    ):

        self.corpus_dir = Path(corpus_dir)

        self.validation_split = validation_split
        self.seed = seed

        if not self.corpus_dir.exists():

            raise FileNotFoundError(
                f"Corpus directory not found: {self.corpus_dir}"
            )

    # ==========================================================
    # Load all text documents
    # ==========================================================

    def load_documents(self):

        documents = []

        for path in sorted(
            self.corpus_dir.rglob("*.txt")
        ):

            try:

                text = path.read_text(
                    encoding="utf-8"
                )

            except UnicodeDecodeError:

                print(
                    f"Skipping non UTF-8 file: {path}"
                )

                continue

            # Remove empty lines while preserving
            # actual document content.

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            text = "\n".join(lines)

            if not text:
                continue

            # Identify document category.

            relative_path = path.relative_to(
                self.corpus_dir
            )

            category = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "root"
            )

            documents.append(
                {
                    "path": path,
                    "category": category,
                    "text": text
                }
            )

        if not documents:

            raise ValueError(
                f"No .txt documents found in "
                f"{self.corpus_dir}"
            )

        return documents

    # ==========================================================
    # Detect Q&A document
    # ==========================================================

    @staticmethod
    def _extract_qa_examples(text):

        """
        Extract independent User/Assistant examples.

        Example:

            User: What is Selenium?
            Assistant: Selenium is ...

            User: What is WebDriver?
            Assistant: WebDriver is ...

        becomes:

            [
                "User: What is Selenium?\\nAssistant: Selenium is ...",
                "User: What is WebDriver?\\nAssistant: WebDriver is ..."
            ]

        Returns None when the document is not a Q&A document.
        """

        pattern = re.compile(
            r"(?ms)"
            r"^User:\s*(.*?)"
            r"^\s*Assistant:\s*(.*?)"
            r"(?=^\s*User:|\Z)"
        )

        matches = pattern.findall(text)

        if not matches:
            return None

        examples = []

        for question, answer in matches:

            question = question.strip()
            answer = answer.strip()

            if not question or not answer:
                continue

            examples.append(
                "User: "
                + question
                + "\nAssistant: "
                + answer
            )

        if not examples:
            return None

        return examples

    # ==========================================================
    # Split Q&A documents into independent examples
    # ==========================================================

    def _expand_documents_into_examples(self, documents):

        """
        Convert Q&A documents into independent examples.

        Non-Q&A documents remain whole documents.

        Each returned item keeps the original category/path
        information so the rest of the pipeline can continue
        working with the same structure.
        """

        expanded = []

        for document in documents:

            qa_examples = self._extract_qa_examples(
                document["text"]
            )

            if qa_examples is None:

                expanded.append(
                    {
                        "path": document["path"],
                        "category": document["category"],
                        "text": document["text"],
                        "source_type": "document"
                    }
                )

                continue

            for index, example in enumerate(
                qa_examples
            ):

                expanded.append(
                    {
                        "path": document["path"],
                        "category": document["category"],
                        "text": example,
                        "source_type": "qa",
                        "example_index": index
                    }
                )

        return expanded

    # ==========================================================
    # Train / validation split
    # ==========================================================

    def train_validation_documents(self):

        documents = self.load_documents()

        # Expand Q&A documents into independent examples.
        #
        # This prevents an entire topic such as Selenium
        # from being placed only in validation.

        examples = self._expand_documents_into_examples(
            documents
        )

        generator = random.Random(
            self.seed
        )

        generator.shuffle(
            examples
        )

        validation_count = max(
            1,
            int(
                len(examples)
                * self.validation_split
            )
        )

        if validation_count >= len(examples):

            validation_count = (
                len(examples) - 1
            )

        validation_documents = examples[
            :validation_count
        ]

        train_documents = examples[
            validation_count:
        ]

        return (
            train_documents,
            validation_documents
        )

    # ==========================================================
    # Backward-compatible text split
    # ==========================================================

    def train_validation_split(self):

        (
            train_documents,
            validation_documents
        ) = self.train_validation_documents()

        train_text = "\n".join(
            document["text"]
            for document in train_documents
        )

        validation_text = "\n".join(
            document["text"]
            for document in validation_documents
        )

        return (
            train_text,
            validation_text
        )