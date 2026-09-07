from pathlib import Path

from tokenizers import Tokenizer


class GenieeTokenizer:

    def __init__(self, tokenizer_path: str | Path):

        self.tokenizer_path = Path(tokenizer_path)

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {self.tokenizer_path}"
            )

        self.tokenizer = Tokenizer.from_file(
            str(self.tokenizer_path)
        )

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.pad_token_id = self.tokenizer.token_to_id(
            self.pad_token
        )

        self.unk_token_id = self.tokenizer.token_to_id(
            self.unk_token
        )

        self.bos_token_id = self.tokenizer.token_to_id(
            self.bos_token
        )

        self.eos_token_id = self.tokenizer.token_to_id(
            self.eos_token
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> list[int]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        token_ids = self.tokenizer.encode(
            text
        ).ids

        if add_special_tokens:
            token_ids = (
                [self.bos_token_id]
                + token_ids
                + [self.eos_token_id]
            )

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True
    ) -> str:

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def token_to_id(self, token: str) -> int | None:

        return self.tokenizer.token_to_id(
            token
        )

    def id_to_token(self, token_id: int) -> str:

        return self.tokenizer.id_to_token(
            token_id
        )