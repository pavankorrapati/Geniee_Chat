from __future__ import annotations

from pathlib import Path
import argparse
import sys

import torch
import torch.nn.functional as F


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model.config import GenieeConfig
from model.geniee_model import GenieeModel
from tokenizer.tokenizer import GenieeTokenizer


# ============================================================
# PATHS
# ============================================================

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "geniee.model"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "geniee_pretrain_best.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Generate text using the pretrained Geniee model."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Software testing is",
        help="Text prompt used to start generation.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Lower = more deterministic.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Keep only the top K tokens before sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )

    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )

    return parser.parse_args()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            f"Tokenizer not found:\n{TOKENIZER_PATH}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print()
    print("Loading tokenizer...")

    tokenizer = GenieeTokenizer(
        model_path=TOKENIZER_PATH
    )

    print(
        f"Vocabulary size : {tokenizer.vocab_size}"
    )

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    saved_config = checkpoint["config"]

    config = GenieeConfig(
        vocab_size=saved_config["vocab_size"],
        max_seq_len=saved_config["max_seq_len"],
        d_model=saved_config["d_model"],
        num_heads=saved_config["num_heads"],
        ffn_hidden_dim=saved_config["ffn_hidden_dim"],
        num_layers=saved_config["num_layers"],
        dropout=saved_config["dropout"],
    )

    model = GenieeModel(
        config=config
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Validation loss  : "
        f"{checkpoint.get('validation_loss', 'unknown')}"
    )

    print(
        f"Device            : {DEVICE}"
    )

    return model, tokenizer, config


# ============================================================
# TOP-K FILTER
# ============================================================

def apply_top_k(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:

    if top_k <= 0:
        return logits

    top_k = min(
        top_k,
        logits.size(-1),
    )

    values, _ = torch.topk(
        logits,
        top_k,
        dim=-1,
    )

    minimum_value = values[..., -1, None]

    logits = torch.where(
        logits < minimum_value,
        torch.full_like(
            logits,
            float("-inf"),
        ),
        logits,
    )

    return logits


# ============================================================
# TOP-P FILTER
# ============================================================

# def apply_top_p(
#     logits: torch.Tensor,
#     top_p: float,
# ) -> torch.Tensor:

#     if top_p >= 1.0:
#         return logits

#     if top_p <= 0.0:
#         return logits

#     sorted_logits, sorted_indices = torch.sort(
#         logits,
#         descending=True,
#         dim=-1,
#     )

#     sorted_probabilities = F.softmax(
#         sorted_logits,
#         dim=-1,
#     )

#     cumulative_probabilities = torch.cumsum(
#         sorted_probabilities,
#         dim=-1,
#     )

#     sorted_remove = (
#         cumulative_probabilities > top_p
#     )

#     # Keep at least one token.
#     sorted_remove[..., 0] = False

#     # Shift the mask so the token that crosses
#     # the threshold is kept.
#     sorted_remove[..., 1:] = (
#         sorted_remove[..., :-1]
#     )

#     remove = torch.zeros_like(
#         sorted_remove
#     )

#     remove.scatter_(
#         -1,
#         sorted_indices,
#         sorted_remove,
#     )

#     logits = logits.masked_fill(
#         remove,
#         float("-inf"),
#     )

#     return logits

# ============================================================
# TOP-P FILTER
# ============================================================

def apply_top_p(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:

    if top_p >= 1.0:
        return logits

    if top_p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    sorted_probabilities = F.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    sorted_remove = (
        cumulative_probabilities > top_p
    )

    # --------------------------------------------------------
    # Keep the first token that crosses the top-p threshold.
    #
    # IMPORTANT:
    # Use clone() here because PyTorch does not allow the
    # overlapping in-place assignment that was previously used.
    # --------------------------------------------------------

    shifted_remove = sorted_remove.clone()

    shifted_remove[..., 1:] = sorted_remove[..., :-1].clone()

    shifted_remove[..., 0] = False

    sorted_remove = shifted_remove

    # --------------------------------------------------------
    # Convert sorted-position mask back to original
    # vocabulary positions.
    # --------------------------------------------------------

    remove = torch.zeros_like(
        sorted_remove
    )

    remove.scatter_(
        -1,
        sorted_indices,
        sorted_remove,
    )

    logits = logits.masked_fill(
        remove,
        float("-inf"),
    )

    return logits


# ============================================================
# SELECT NEXT TOKEN
# ============================================================

def select_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
) -> torch.Tensor:

    if greedy:

        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    if temperature <= 0:

        raise ValueError(
            "Temperature must be greater than 0."
        )

    logits = logits / temperature

    logits = apply_top_k(
        logits,
        top_k,
    )

    logits = apply_top_p(
        logits,
        top_p,
    )

    probabilities = F.softmax(
        logits,
        dim=-1,
    )

    next_token = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return next_token


# ============================================================
# GENERATE
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    config,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
):

    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    token_ids = tokenizer.encode(
        prompt
    )

    if len(token_ids) == 0:

        raise ValueError(
            "Prompt produced zero tokens."
        )

    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    for _ in range(max_new_tokens):

        # Keep only the most recent context if necessary.
        if input_ids.size(1) > config.max_seq_len:

            input_ids = input_ids[
                :, -config.max_seq_len:
            ]

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            input_ids
        )

        # ----------------------------------------------------
        # Last token logits
        # ----------------------------------------------------

        next_token_logits = logits[
            :, -1, :
        ]

        # ----------------------------------------------------
        # Select token
        # ----------------------------------------------------

        next_token = select_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )

        # ----------------------------------------------------
        # Append token
        # ----------------------------------------------------

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    generated_ids = (
        input_ids[0]
        .detach()
        .cpu()
        .tolist()
    )

    generated_text = tokenizer.decode(
        generated_ids
    )

    return generated_text


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    print()
    print("=" * 70)
    print("GENIEE TEXT GENERATION")
    print("=" * 70)

    print()
    print(f"Device : {DEVICE}")

    print()
    print("-" * 70)
    print("LOADING MODEL")
    print("-" * 70)

    model, tokenizer, config = load_model()

    print()
    print("-" * 70)
    print("GENERATION CONFIGURATION")
    print("-" * 70)

    print()
    print(f"Prompt          : {args.prompt}")
    print(f"Max new tokens  : {args.max_new_tokens}")
    print(f"Temperature     : {args.temperature}")
    print(f"Top K           : {args.top_k}")
    print(f"Top P           : {args.top_p}")
    print(f"Greedy          : {args.greedy}")

    print()
    print("-" * 70)
    print("GENERATING")
    print("-" * 70)

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )

    print()
    print("=" * 70)
    print("GENERATED TEXT")
    print("=" * 70)

    print()
    print(generated_text)

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()