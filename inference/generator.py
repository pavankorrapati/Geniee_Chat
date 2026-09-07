from __future__ import annotations

import torch
import torch.nn.functional as F


class GenieeGenerator:
    """
    Autoregressive text generator for Geniee.

    Responsibilities:
        - Generate tokens from a trained Geniee model
        - Temperature scaling
        - Top-k filtering
        - Top-p / nucleus sampling
        - Repetition penalty
        - EOS stopping
        - Attention-mask aware generation
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()

    # ======================================================
    # Temperature
    # ======================================================

    @staticmethod
    def apply_temperature(
        logits,
        temperature,
    ):
        """
        Apply temperature scaling.

        temperature < 1:
            More deterministic

        temperature > 1:
            More random
        """

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0."
            )

        return logits / temperature

    # ======================================================
    # Repetition penalty
    # ======================================================

    @staticmethod
    def apply_repetition_penalty(
        logits,
        generated_ids,
        repetition_penalty,
    ):
        """
        Penalize tokens that already appeared.

        1.0 = disabled
        >1.0 = stronger penalty
        """

        if repetition_penalty <= 0:
            raise ValueError(
                "repetition_penalty must be greater than 0."
            )

        if repetition_penalty == 1.0:
            return logits

        for token_id in set(generated_ids):

            score = logits[token_id]

            if score < 0:
                logits[token_id] = (
                    score * repetition_penalty
                )

            else:
                logits[token_id] = (
                    score / repetition_penalty
                )

        return logits

    # ======================================================
    # Top-k
    # ======================================================

    @staticmethod
    def apply_top_k(
        logits,
        top_k,
    ):
        """
        Keep only the top-k highest-logit tokens.

        top_k <= 0:
            Disabled
        """

        if top_k <= 0:
            return logits

        top_k = min(
            top_k,
            logits.size(-1),
        )

        threshold = torch.topk(
            logits,
            top_k,
        ).values[-1]

        logits[
            logits < threshold
        ] = -float("inf")

        return logits

    # ======================================================
    # Top-p
    # ======================================================

    @staticmethod
    def apply_top_p(
        logits,
        top_p,
    ):
        """
        Nucleus sampling.

        Keeps the smallest set of tokens whose
        cumulative probability reaches top_p.
        """

        if top_p >= 1.0:
            return logits

        if top_p <= 0:
            raise ValueError(
                "top_p must be greater than 0."
            )

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        sorted_probabilities = F.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        sorted_remove_mask = (
            cumulative_probabilities > top_p
        )

        shifted_remove_mask = torch.zeros_like(
            sorted_remove_mask
        )

        shifted_remove_mask[1:] = (
            sorted_remove_mask[:-1].clone()
        )

        sorted_remove_mask = shifted_remove_mask

        # Always keep the highest-probability token.

        sorted_remove_mask[0] = False

        remove_indices = sorted_indices[
            sorted_remove_mask
        ]

        logits[remove_indices] = -float("inf")

        return logits

    # ======================================================
    # Greedy selection
    # ======================================================

    @staticmethod
    def select_greedy_token(
        logits,
    ):
        """
        Select the highest-probability token.

        Used for deterministic debugging.
        """

        return torch.argmax(
            logits
        ).item()

    # ======================================================
    # Sample next token
    # ======================================================

    def sample_next_token(
        self,
        logits,
        generated_ids,
        temperature=0.8,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.1,
    ):
        """
        Apply generation controls and sample
        the next token.
        """

        logits = logits.clone()

        logits = self.apply_repetition_penalty(
            logits=logits,
            generated_ids=generated_ids,
            repetition_penalty=repetition_penalty,
        )

        logits = self.apply_temperature(
            logits=logits,
            temperature=temperature,
        )

        logits = self.apply_top_k(
            logits=logits,
            top_k=top_k,
        )

        logits = self.apply_top_p(
            logits=logits,
            top_p=top_p,
        )

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

        if not torch.isfinite(
            probabilities
        ).all():
            raise RuntimeError(
                "Invalid probabilities encountered "
                "during generation."
            )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        return next_token.item()

    # ======================================================
    # Generate
    # ======================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.1,
        do_sample=True,
    ):
        """
        Generate tokens autoregressively.

        Parameters
        ----------
        input_ids:
            List of token IDs.

        max_new_tokens:
            Maximum number of tokens to generate.

        do_sample:
            False -> greedy deterministic generation
            True  -> temperature/top-k/top-p sampling
        """

        if not input_ids:
            raise ValueError(
                "input_ids cannot be empty."
            )

        generated_ids = list(
            input_ids
        )

        eos_id = self.tokenizer.eos_id

        max_context_length = (
            self.model.config.max_seq_len
        )

        for _ in range(
            max_new_tokens
        ):

            # ------------------------------------------------
            # Respect model context length
            # ------------------------------------------------

            context_ids = generated_ids[
                -max_context_length:
            ]

            input_tensor = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=self.device,
            )

            # ------------------------------------------------
            # Attention mask
            #
            # During generation all context tokens are
            # real tokens, so mask = 1.
            # ------------------------------------------------

            attention_mask = torch.ones(
                input_tensor.shape,
                dtype=torch.long,
                device=self.device,
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = self.model(
                input_tensor,
                attention_mask=attention_mask,
            )

            # ------------------------------------------------
            # Last position logits
            # ------------------------------------------------

            next_token_logits = (
                logits[0, -1, :]
            )

            # ------------------------------------------------
            # Select next token
            # ------------------------------------------------

            if do_sample:

                next_token_id = (
                    self.sample_next_token(
                        logits=next_token_logits,
                        generated_ids=generated_ids,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=(
                            repetition_penalty
                        ),
                    )
                )

            else:

                next_token_id = (
                    self.select_greedy_token(
                        next_token_logits
                    )
                )

            # ------------------------------------------------
            # Append token
            # ------------------------------------------------

            generated_ids.append(
                next_token_id
            )

            # ------------------------------------------------
            # EOS stopping
            # ------------------------------------------------

            if next_token_id == eos_id:
                break

        return generated_ids

    # ======================================================
    # Generate text
    # ======================================================

    @torch.no_grad()
    def generate_text(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.1,
        do_sample=True,
    ):
        """
        Generate text and decode it.
        """

        generated_ids = self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=(
                repetition_penalty
            ),
            do_sample=do_sample,
        )

        return self.tokenizer.decode(
            generated_ids
        )