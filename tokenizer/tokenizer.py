from pathlib import Path

import sentencepiece as spm


class GenieeTokenizer:
    """
    Wrapper around the SentencePiece tokenizer used by Geniee.
    """

    def __init__(self, model_path):

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Tokenizer model not found: "
                f"{self.model_path}"
            )

        self.processor = (
            spm.SentencePieceProcessor()
        )

        self.processor.load(
            str(self.model_path)
        )

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):
        """
        Convert text into token IDs.
        """

        token_ids = self.processor.encode(
            text,
            out_type=int
        )

        if add_bos:

            token_ids.insert(
                0,
                self.bos_id
            )

        if add_eos:

            token_ids.append(
                self.eos_id
            )

        return token_ids

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------

    def decode(self, token_ids):
        """
        Convert token IDs back into text.
        """

        return self.processor.decode(
            token_ids
        )

    # --------------------------------------------------
    # Vocabulary size
    # --------------------------------------------------

    @property
    def vocab_size(self):

        return self.processor.get_piece_size()

    # --------------------------------------------------
    # Special token IDs
    # --------------------------------------------------

    @property
    def pad_id(self):

        return self.processor.pad_id()

    @property
    def unk_id(self):

        return self.processor.unk_id()

    @property
    def bos_id(self):

        return self.processor.bos_id()

    @property
    def eos_id(self):

        return self.processor.eos_id()