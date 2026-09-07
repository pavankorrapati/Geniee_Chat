# from pathlib import Path
# import sys

# import torch


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

# from model.config import GenieeConfig
# from model.geniee_model import GenieeModel
# from tokenizer.tokenizer import GenieeTokenizer


# # ==========================================================
# # Geniee Generator
# # ==========================================================

# class GenieeGenerator:
#     """
#     Generates text using a trained Geniee language model.
#     """

#     def __init__(
#         self,
#         model,
#         tokenizer,
#         device="cpu"
#     ):

#         self.device = torch.device(
#             device
#         )

#         self.model = model.to(
#             self.device
#         )

#         self.tokenizer = tokenizer

#         self.model.eval()

#     # ======================================================
#     # Generate token IDs
#     # ======================================================

#     @torch.no_grad()
#     def generate_tokens(
#         self,
#         input_ids,
#         max_new_tokens=50
#     ):
#         """
#         Generate new tokens from existing token IDs.

#         Uses greedy decoding:

#             next_token = argmax(logits)
#         """

#         input_ids = input_ids.to(
#             self.device
#         )

#         for _ in range(
#             max_new_tokens
#         ):

#             # --------------------------------------------------
#             # Keep sequence within model context length
#             # --------------------------------------------------

#             max_seq_len = (
#                 self.model.config.max_seq_len
#             )

#             if input_ids.size(1) > max_seq_len:

#                 input_ids = (
#                     input_ids[:, -max_seq_len:]
#                 )

#             # --------------------------------------------------
#             # Forward pass
#             # --------------------------------------------------

#             logits = self.model(
#                 input_ids
#             )

#             # --------------------------------------------------
#             # Get logits for final position
#             # --------------------------------------------------

#             next_token_logits = (
#                 logits[:, -1, :]
#             )

#             # --------------------------------------------------
#             # Greedy selection
#             # --------------------------------------------------

#             next_token = torch.argmax(
#                 next_token_logits,
#                 dim=-1,
#                 keepdim=True
#             )

#             # --------------------------------------------------
#             # Append token
#             # --------------------------------------------------

#             input_ids = torch.cat(
#                 [
#                     input_ids,
#                     next_token
#                 ],
#                 dim=1
#             )

#             # --------------------------------------------------
#             # Stop at EOS
#             # --------------------------------------------------

#             eos_id = getattr(
#                 self.tokenizer,
#                 "eos_id",
#                 None
#             )

#             if (
#                 eos_id is not None
#                 and torch.all(
#                     next_token == eos_id
#                 )
#             ):

#                 break

#         return input_ids

#     # ======================================================
#     # Generate text
#     # ======================================================

#     def generate(
#         self,
#         prompt,
#         max_new_tokens=50
#     ):
#         """
#         Convert prompt to tokens, generate new tokens,
#         and decode the result back into text.
#         """

#         # --------------------------------------------------
#         # Encode prompt
#         # --------------------------------------------------

#         token_ids = self.tokenizer.encode(
#             prompt,
#             add_bos=True,
#             add_eos=False
#         )

#         input_ids = torch.tensor(
#             [token_ids],
#             dtype=torch.long,
#             device=self.device
#         )

#         # --------------------------------------------------
#         # Generate
#         # --------------------------------------------------

#         output_ids = self.generate_tokens(
#             input_ids,
#             max_new_tokens=max_new_tokens
#         )

#         # --------------------------------------------------
#         # Convert back to text
#         # --------------------------------------------------

#         generated_token_ids = (
#             output_ids[0]
#             .detach()
#             .cpu()
#             .tolist()
#         )

#         generated_text = (
#             self.tokenizer.decode(
#                 generated_token_ids
#             )
#         )

#         return generated_text


# # ==========================================================
# # Load trained Geniee model
# # ==========================================================

# def load_geniee(
#     checkpoint_path
# ):

#     checkpoint_path = Path(
#         checkpoint_path
#     )

#     if not checkpoint_path.exists():

#         raise FileNotFoundError(
#             "Checkpoint not found:\n"
#             f"{checkpoint_path}"
#         )

#     # ------------------------------------------------------
#     # Tokenizer
#     # ------------------------------------------------------

#     tokenizer_path = (
#         PROJECT_ROOT
#         / "tokenizer"
#         / "artifacts"
#         / "geniee.model"
#     )

#     if not tokenizer_path.exists():

#         raise FileNotFoundError(
#             "Tokenizer not found:\n"
#             f"{tokenizer_path}"
#         )

#     tokenizer = GenieeTokenizer(
#         model_path=tokenizer_path
#     )

#     # ------------------------------------------------------
#     # Configuration
#     # ------------------------------------------------------

#     config = GenieeConfig(
#         vocab_size=tokenizer.vocab_size,
#         max_seq_len=8
#     )

#     # ------------------------------------------------------
#     # Model
#     # ------------------------------------------------------

#     model = GenieeModel(
#         config
#     )

#     # ------------------------------------------------------
#     # Load checkpoint
#     # ------------------------------------------------------

#     checkpoint = torch.load(
#         checkpoint_path,
#         map_location="cpu"
#     )

#     model.load_state_dict(
#         checkpoint[
#             "model_state_dict"
#         ]
#     )

#     # ------------------------------------------------------
#     # Generator
#     # ------------------------------------------------------

#     generator = GenieeGenerator(
#         model=model,
#         tokenizer=tokenizer,
#         device="cpu"
#     )

#     return generator


# # ==========================================================
# # Main
# # ==========================================================

# def main():

#     checkpoint_path = (
#         PROJECT_ROOT
#         / "checkpoints"
#         / "geniee_epoch_5.pt"
#     )

#     print()
#     print(
#         "=" * 60
#     )

#     print(
#         "GENIEE TEXT GENERATION"
#     )

#     print(
#         "=" * 60
#     )

#     print(
#         f"Checkpoint: {checkpoint_path}"
#     )

#     # ------------------------------------------------------
#     # Load model
#     # ------------------------------------------------------

#     generator = load_geniee(
#         checkpoint_path
#     )

#     # ------------------------------------------------------
#     # Prompt
#     # ------------------------------------------------------

#     prompt = (
#         "Geniee is"
#     )

#     print()
#     print(
#         f"Prompt: {prompt}"
#     )

#     # ------------------------------------------------------
#     # Generate
#     # ------------------------------------------------------

#     generated_text = (
#         generator.generate(
#             prompt,
#             max_new_tokens=30
#         )
#     )

#     print()
#     print(
#         "Generated text:"
#     )

#     print(
#         generated_text
#     )

#     print()
#     print(
#         "=" * 60
#     )


# # ==========================================================
# # Entry point
# # ==========================================================

# if __name__ == "__main__":

#     main()

from pathlib import Path
import sys

import torch


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

from model.config import GenieeConfig
from model.geniee_model import GenieeModel
from tokenizer.tokenizer import GenieeTokenizer


# ==========================================================
# Geniee Generator
# ==========================================================

class GenieeGenerator:
    """
    Text generation engine for Geniee.

    Supported decoding strategies:

        1. Greedy decoding
        2. Temperature sampling
        3. Top-K sampling
        4. Top-P / nucleus sampling
        5. Repetition penalty
    """

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu"
    ):

        self.device = torch.device(
            device
        )

        self.model = model.to(
            self.device
        )

        self.tokenizer = tokenizer

        self.model.eval()

    # ======================================================
    # Apply repetition penalty
    # ======================================================

    def _apply_repetition_penalty(
        self,
        logits,
        input_ids,
        repetition_penalty
    ):
        """
        Penalize tokens that have already appeared
        in the generated sequence.
        """

        if repetition_penalty <= 0:

            raise ValueError(
                "repetition_penalty must be > 0"
            )

        if repetition_penalty == 1.0:

            return logits

        for token_id in torch.unique(
            input_ids
        ):

            token_id = token_id.item()

            if logits[0, token_id] < 0:

                logits[0, token_id] *= (
                    repetition_penalty
                )

            else:

                logits[0, token_id] /= (
                    repetition_penalty
                )

        return logits

    # ======================================================
    # Apply temperature
    # ======================================================

    def _apply_temperature(
        self,
        logits,
        temperature
    ):
        """
        Temperature controls randomness.

        Lower temperature:
            More deterministic.

        Higher temperature:
            More random.
        """

        if temperature <= 0:

            raise ValueError(
                "temperature must be > 0"
            )

        if temperature == 1.0:

            return logits

        return logits / temperature

    # ======================================================
    # Apply top-k
    # ======================================================

    def _apply_top_k(
        self,
        logits,
        top_k
    ):
        """
        Keep only the top K most probable tokens.
        """

        if top_k is None:

            return logits

        if top_k <= 0:

            return logits

        vocabulary_size = logits.size(-1)

        top_k = min(
            top_k,
            vocabulary_size
        )

        values, _ = torch.topk(
            logits,
            top_k,
            dim=-1
        )

        minimum_value = values[
            :,
            -1
        ].unsqueeze(-1)

        logits = torch.where(
            logits < minimum_value,
            torch.full_like(
                logits,
                float("-inf")
            ),
            logits
        )

        return logits

    # ======================================================
    # Apply top-p
    # ======================================================

    def _apply_top_p(
        self,
        logits,
        top_p
    ):
        """
        Nucleus / Top-P sampling.

        Keeps the smallest set of tokens whose cumulative
        probability reaches the top_p threshold.
        """

        # --------------------------------------------------
        # No Top-P filtering requested
        # --------------------------------------------------

        if top_p is None:

            return logits

        # --------------------------------------------------
        # Validate Top-P
        # --------------------------------------------------

        if top_p <= 0 or top_p > 1:

            raise ValueError(
                "top_p must be between 0 and 1"
            )

        # --------------------------------------------------
        # No filtering when top_p == 1
        # --------------------------------------------------

        if top_p == 1.0:

            return logits

        # --------------------------------------------------
        # Sort logits from highest to lowest
        # --------------------------------------------------

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1
        )

        # --------------------------------------------------
        # Convert logits to probabilities
        # --------------------------------------------------

        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1
        )

        # --------------------------------------------------
        # Calculate cumulative probabilities
        # --------------------------------------------------

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1
        )

        # --------------------------------------------------
        # Identify tokens beyond Top-P
        # --------------------------------------------------

        sorted_remove = (
            cumulative_probabilities > top_p
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT perform overlapping in-place assignment:
        #
        # sorted_remove[:, 1:] = sorted_remove[:, :-1]
        #
        # Clone the source first.
        # --------------------------------------------------

        shifted_remove = sorted_remove.clone()

        shifted_remove[:, 1:] = (
            sorted_remove[:, :-1]
        )

        # First token must always remain available
        shifted_remove[:, 0] = False

        # --------------------------------------------------
        # Convert sorted mask back to original vocabulary
        # positions.
        # --------------------------------------------------

        remove_indices = torch.zeros_like(
            logits,
            dtype=torch.bool
        )

        remove_indices.scatter_(
            dim=-1,
            index=sorted_indices,
            src=shifted_remove
        )

        # --------------------------------------------------
        # Remove filtered tokens
        # --------------------------------------------------

        filtered_logits = logits.masked_fill(
            remove_indices,
            float("-inf")
        )

        return filtered_logits

    def _suppress_special_tokens(
        self,
        logits
    ):
        """
        Prevent invalid/special tokens from being generated
        during normal text generation.

        Allowed:
            EOS

        Suppressed:
            UNK
            BOS
        """

        unk_id = getattr(
            self.tokenizer,
            "unk_id",
            None
        )

        bos_id = getattr(
            self.tokenizer,
            "bos_id",
            None
        )

        if unk_id is not None:

            logits[:, unk_id] = float("-inf")

        if bos_id is not None:

            logits[:, bos_id] = float("-inf")

        return logits

    # ======================================================
    # Select next token
    # ======================================================

    def _select_next_token(
        self,
        logits,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=True
    ):
        """
        Select the next token from logits.
        """

        # --------------------------------------------------
        # Temperature
        # --------------------------------------------------

        logits = self._apply_temperature(
            logits,
            temperature
        )

        # --------------------------------------------------
        # Top-K
        # --------------------------------------------------

        logits = self._apply_top_k(
            logits,
            top_k
        )

        # --------------------------------------------------
        # Top-P
        # --------------------------------------------------

        logits = self._apply_top_p(
            logits,
            top_p
        )

        # --------------------------------------------------
        # Greedy
        # --------------------------------------------------

        if not do_sample:

            return torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        # --------------------------------------------------
        # Sampling
        # --------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        return next_token

    # ======================================================
    # Generate tokens
    # ======================================================

    @torch.no_grad()
    def generate_tokens(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        do_sample=True
    ):
        """
        Generate new token IDs.

        Parameters
        ----------
        input_ids:
            Tensor of shape [batch, sequence].

        max_new_tokens:
            Maximum number of tokens to generate.

        temperature:
            Controls randomness.

        top_k:
            Restrict sampling to top K tokens.

        top_p:
            Restrict sampling to cumulative probability.

        repetition_penalty:
            Penalize previously generated tokens.

        do_sample:
            False = greedy decoding.
            True = probabilistic sampling.
        """

        if max_new_tokens <= 0:

            raise ValueError(
                "max_new_tokens must be > 0"
            )

        input_ids = input_ids.to(
            self.device
        )

        for _ in range(
            max_new_tokens
        ):

            # --------------------------------------------------
            # Respect model context window
            # --------------------------------------------------

            max_seq_len = (
                self.model.config.max_seq_len
            )

            model_input = input_ids

            if (
                model_input.size(1)
                > max_seq_len
            ):

                model_input = (
                    model_input[
                        :,
                        -max_seq_len:
                    ]
                )

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------

            logits = self.model(
                model_input
            )

            # --------------------------------------------------
            # Last token logits
            # --------------------------------------------------

            next_token_logits = (
                logits[:, -1, :]
            )
            next_token_logits = (self._suppress_special_tokens(next_token_logits))

            # --------------------------------------------------
            # Repetition penalty
            # --------------------------------------------------

            next_token_logits = (
                self._apply_repetition_penalty(
                    next_token_logits,
                    input_ids,
                    repetition_penalty
                )
            )

            # --------------------------------------------------
            # Select token
            # --------------------------------------------------

            next_token = (
                self._select_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample
                )
            )

            # --------------------------------------------------
            # Append
            # --------------------------------------------------

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

            # --------------------------------------------------
            # EOS
            # --------------------------------------------------

            eos_id = getattr(
                self.tokenizer,
                "eos_id",
                None
            )

            if (
                eos_id is not None
                and torch.all(
                    next_token == eos_id
                )
            ):

                break

        return input_ids

    # ======================================================
    # Generate text
    # ======================================================

    # def generate(
    #     self,
    #     prompt,
    #     max_new_tokens=50,
    #     temperature=1.0,
    #     top_k=None,
    #     top_p=None,
    #     repetition_penalty=1.0,
    #     do_sample=True
    # ):
    #     """
    #     Generate text from a string prompt.
    #     """

    #     # --------------------------------------------------
    #     # Encode prompt
    #     # --------------------------------------------------

    #     token_ids = self.tokenizer.encode(
    #         prompt,
    #         add_bos=True,
    #         add_eos=False
    #     )

    #     input_ids = torch.tensor(
    #         [token_ids],
    #         dtype=torch.long,
    #         device=self.device
    #     )

    #     # --------------------------------------------------
    #     # Generate
    #     # --------------------------------------------------

    #     output_ids = self.generate_tokens(
    #         input_ids=input_ids,
    #         max_new_tokens=max_new_tokens,
    #         temperature=temperature,
    #         top_k=top_k,
    #         top_p=top_p,
    #         repetition_penalty=repetition_penalty,
    #         do_sample=do_sample
    #     )

    #     # --------------------------------------------------
    #     # Decode
    #     # --------------------------------------------------

    #     generated_token_ids = (
    #         output_ids[0]
    #         .detach()
    #         .cpu()
    #         .tolist()
    #     )

    #     generated_text = (
    #         self.tokenizer.decode(
    #             generated_token_ids
    #         )
    #     )

    #     return generated_text
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        do_sample=True,
        return_full_text=True
    ):
        """
        Generate text from a string prompt.

        Parameters
        ----------
        prompt:
            Input text.

        max_new_tokens:
            Number of new tokens to generate.

        temperature:
            Controls randomness.

        top_k:
            Restricts sampling to the top K tokens.

        top_p:
            Nucleus sampling threshold.

        repetition_penalty:
            Penalizes previously generated tokens.

        do_sample:
            True  -> sampling
            False -> greedy decoding

        return_full_text:
            True  -> return prompt + generated text
            False -> return only newly generated text.
        """

        # ------------------------------------------------------
        # Validate prompt
        # ------------------------------------------------------

        if not isinstance(
            prompt,
            str
        ):

            raise TypeError(
                "prompt must be a string"
            )

        prompt = prompt.strip()

        if not prompt:

            return ""

        # ------------------------------------------------------
        # Encode prompt
        # ------------------------------------------------------

        token_ids = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False
        )

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device
        )

        # ------------------------------------------------------
        # Remember prompt length
        # ------------------------------------------------------

        prompt_length = (
            input_ids.shape[1]
        )

        # ------------------------------------------------------
        # Generate
        # ------------------------------------------------------

        output_ids = self.generate_tokens(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )

        # ------------------------------------------------------
        # Extract only newly generated tokens
        # ------------------------------------------------------

        generated_ids = (
            output_ids[
                0,
                prompt_length:
            ]
            .detach()
            .cpu()
            .tolist()
        )

        # ------------------------------------------------------
        # Decode generated tokens
        # ------------------------------------------------------

        generated_text = (
            self.tokenizer.decode(
                generated_ids
            )
        )

        generated_text = (
            generated_text.strip()
        )

        # ------------------------------------------------------
        # Return
        # ------------------------------------------------------

        if return_full_text:

            full_text = (
                self.tokenizer.decode(
                    output_ids[0]
                    .detach()
                    .cpu()
                    .tolist()
                )
            )

            return full_text.strip()

        return generated_text


# ==========================================================
# Load Geniee
# ==========================================================

def load_geniee(
    checkpoint_path
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    # ------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------

    tokenizer_path = (
        PROJECT_ROOT
        / "tokenizer"
        / "artifacts"
        / "geniee.model"
    )

    if not tokenizer_path.exists():

        raise FileNotFoundError(
            "Tokenizer not found:\n"
            f"{tokenizer_path}"
        )

    tokenizer = GenieeTokenizer(
        model_path=tokenizer_path
    )

    # ------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------

    config = GenieeConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = GenieeModel(
        config
    )

    # ------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ------------------------------------------------------
    # Generator
    # ------------------------------------------------------

    generator = GenieeGenerator(
        model=model,
        tokenizer=tokenizer,
        device="cpu"
    )

    return generator


# ==========================================================
# Demo
# ==========================================================

def main():

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "geniee_epoch_5.pt"
    )

    generator = load_geniee(
        checkpoint_path
    )

    prompt = "Geniee is"

    print()
    print("=" * 60)
    print("GENIEE GENERATION")
    print("=" * 60)

    print()
    print("Prompt:")
    print(prompt)

    # ------------------------------------------------------
    # Greedy
    # ------------------------------------------------------

    greedy_text = generator.generate(
        prompt,
        max_new_tokens=20,
        do_sample=False
    )

    print()
    print("Greedy:")
    print(greedy_text)

    # ------------------------------------------------------
    # Sampling
    # ------------------------------------------------------

    sampled_text = generator.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

    print()
    print("Sampling:")
    print(sampled_text)

    print()
    print("=" * 60)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()