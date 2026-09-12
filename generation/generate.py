from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import GenieeConfig
from model.geniee_model import GenieeModel
from tokenizer.tokenizer import GenieeTokenizer


class GenieeGenerator:
    """Reliable autoregressive generation for Geniee."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model.eval()

    @staticmethod
    def _apply_repetition_penalty(logits, token_ids, penalty):
        if penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if penalty == 1.0:
            return logits
        for token_id in set(token_ids):
            score = logits[0, token_id]
            logits[0, token_id] = score * penalty if score < 0 else score / penalty
        return logits

    @staticmethod
    def _apply_top_k(logits, top_k):
        if not top_k or top_k <= 0:
            return logits
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _apply_top_p(logits, top_p):
        if top_p is None or top_p >= 1.0:
            return logits
        if top_p <= 0:
            raise ValueError("top_p must be > 0")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        shifted = torch.zeros_like(remove)
        shifted[..., 1:] = remove[..., :-1].clone()
        shifted[..., 0] = False
        original_remove = torch.zeros_like(remove)
        original_remove.scatter_(-1, sorted_indices, shifted)
        return logits.masked_fill(original_remove, float("-inf"))

    def _suppress_special_tokens(self, logits):
        logits = logits.clone()
        for name in ("unk_id", "bos_id"):
            token_id = getattr(self.tokenizer, name, None)
            if token_id is not None and token_id >= 0:
                logits[:, token_id] = float("-inf")
        return logits

    @staticmethod
    def _sample(logits, temperature, top_k, top_p, do_sample):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = logits / temperature
        logits = GenieeGenerator._apply_top_k(logits, top_k)
        logits = GenieeGenerator._apply_top_p(logits, top_p)
        if not do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all():
            raise RuntimeError("Invalid probabilities encountered during generation")
        return torch.multinomial(probs, 1)

    @torch.no_grad()
    def generate_tokens(
        self,
        input_ids,
        max_new_tokens=80,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.05,
        do_sample=True,
        stop_on_role_markers=True,
    ):
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("input_ids must have shape [1, sequence_length]")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

        generated = input_ids.to(self.device)
        prompt_length = generated.size(1)
        eos_id = getattr(self.tokenizer, "eos_id", None)
        stop_markers = ("<|user|>", "<|system|>", "<|assistant|>")

        for _ in range(max_new_tokens):
            context = generated[:, -self.model.config.max_seq_len:]
            logits = self.model(context)[:, -1, :]
            logits = self._suppress_special_tokens(logits)

            # Penalize only generated tokens, not the user's prompt.
            generated_only = generated[0, prompt_length:].detach().cpu().tolist()
            logits = self._apply_repetition_penalty(logits, generated_only, repetition_penalty)

            next_token = self._sample(logits, temperature, top_k, top_p, do_sample)
            generated = torch.cat([generated, next_token], dim=1)
            next_id = next_token.item()

            if eos_id is not None and next_id == eos_id:
                break

            if stop_on_role_markers and generated.size(1) > prompt_length:
                text = self.tokenizer.decode(generated[0, prompt_length:].detach().cpu().tolist())
                if any(marker in text for marker in stop_markers):
                    break

        return generated

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=80,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.05,
        do_sample=True,
        return_full_text=False,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        # IMPORTANT: do not strip the prompt's trailing newline. The SFT
        # dataset trains on <|assistant|>\\n, so inference must reproduce it.
        token_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        prompt_length = input_ids.size(1)

        output_ids = self.generate_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

        full = self.tokenizer.decode(output_ids[0].detach().cpu().tolist()).strip()
        new = self.tokenizer.decode(output_ids[0, prompt_length:].detach().cpu().tolist()).strip()
        if return_full_text:
            return full
        return new


def load_geniee(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    tokenizer_path = PROJECT_ROOT / "tokenizer" / "artifacts" / "geniee.model"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found:\n{tokenizer_path}")

    tokenizer = GenieeTokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved = checkpoint.get("config", {})

    config = GenieeConfig(
        vocab_size=int(saved.get("vocab_size", tokenizer.vocab_size)),
        max_seq_len=int(saved.get("max_seq_len", 256)),
        d_model=int(saved.get("d_model", 512)),
        num_heads=int(saved.get("num_heads", 8)),
        ffn_hidden_dim=int(saved.get("ffn_hidden_dim", 2048)),
        num_layers=int(saved.get("num_layers", 8)),
        dropout=float(saved.get("dropout", 0.0)),
    )

    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Checkpoint vocabulary ({config.vocab_size}) does not match tokenizer ({tokenizer.vocab_size})."
        )

    model = GenieeModel(config)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return GenieeGenerator(model, tokenizer, device or ("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    checkpoint = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"
    generator = load_geniee(checkpoint)
    prompt = "<|system|>\\nYou are Geniee, a helpful AI assistant.\\n<|user|>\\nWhat is Python?\\n<|assistant|>\\n"
    print(generator.generate(prompt, max_new_tokens=60, do_sample=False))
