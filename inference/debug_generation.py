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
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# Imports
# ==========================================================

from tokenizer.tokenizer import GenieeTokenizer
from model.config import GenieeConfig
from model.geniee_model import GenieeModel


# ==========================================================
# Paths
# ==========================================================

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "geniee.model"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "geniee_best.pt"
)


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device("cpu")

MAX_SEQ_LEN = 128


# ==========================================================
# Tokenizer
# ==========================================================

tokenizer = GenieeTokenizer(
    model_path=TOKENIZER_PATH
)


# ==========================================================
# Model
# ==========================================================

config = GenieeConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=MAX_SEQ_LEN,
    d_model=512,
    num_heads=8,
    ffn_hidden_dim=2048,
    num_layers=8,
    dropout=0.1
)

model = GenieeModel(config)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()


# ==========================================================
# Diagnostic
# ==========================================================

@torch.no_grad()
def diagnose(prompt):

    formatted_prompt = (
        "User: "
        + prompt.strip()
        + "\nAssistant:"
    )

    token_ids = tokenizer.encode(
        formatted_prompt,
        add_bos=True,
        add_eos=False
    )

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    logits = model(
        input_ids[:, -MAX_SEQ_LEN:]
    )

    next_token_logits = logits[:, -1, :]

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1
    )

    top_values, top_indices = torch.topk(
        probabilities,
        k=10,
        dim=-1
    )

    print()
    print("=" * 60)
    print("GENERATION DIAGNOSTIC")
    print("=" * 60)

    print()
    print("Prompt:")
    print(formatted_prompt)

    print()
    print("Top 10 next-token predictions:")
    print("-" * 60)

    for rank in range(10):

        token_id = top_indices[0, rank].item()
        probability = top_values[0, rank].item()

        token_text = tokenizer.decode(
            [token_id]
        )

        print(
            f"{rank + 1:2d}. "
            f"ID={token_id:4d} "
            f"PROB={probability:.6f} "
            f"TOKEN={repr(token_text)}"
        )

    print()
    print("=" * 60)


# ==========================================================
# Tests
# ==========================================================

if __name__ == "__main__":

    diagnose("What is Selenium?")
    diagnose("What is software testing?")
    diagnose("What is API testing?")
    diagnose("What is Playwright?")