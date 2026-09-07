from pathlib import Path
import sys
import json
import torch

# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import GenieeConfig
from model.geniee_model import GenieeModel
from tokenizer.tokenizer import GenieeTokenizer
from inference.generator import GenieeGenerator


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
    / "geniee_sft_best.pt"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.jsonl"
)


# ==========================================================
# Load training example
# ==========================================================

with TRAIN_FILE.open(
    "r",
    encoding="utf-8"
) as f:
    record = json.loads(f.readline())


messages = record["messages"]

system_message = next(
    m["content"]
    for m in messages
    if m["role"] == "system"
)

user_message = next(
    m["content"]
    for m in messages
    if m["role"] == "user"
)

expected_answer = next(
    m["content"]
    for m in messages
    if m["role"] == "assistant"
)


# ==========================================================
# Prompt
# ==========================================================

prompt = (
    "<|system|>\n"
    + system_message.strip()
    + "\n"
    + "<|user|>\n"
    + user_message.strip()
    + "\n"
    + "<|assistant|>\n"
)


# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("GENIEE TRAINING-EXAMPLE MEMORIZATION TEST")
print("=" * 70)

print(f"\nDevice    : {device}")
print(f"Tokenizer : {TOKENIZER_PATH}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


# ==========================================================
# Tokenizer
# ==========================================================

tokenizer = GenieeTokenizer(
    str(TOKENIZER_PATH)
)

print(f"Vocabulary: {tokenizer.vocab_size}")


# ==========================================================
# Model
# ==========================================================

config = GenieeConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=128,
    d_model=512,
    num_heads=8,
    ffn_hidden_dim=2048,
    num_layers=8,
    dropout=0.1,
)

model = GenieeModel(config).to(device)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()


# ==========================================================
# Generator
# ==========================================================

generator = GenieeGenerator(
    model=model,
    tokenizer=tokenizer,
    device=device,
)


# ==========================================================
# Encode prompt
# ==========================================================

prompt_ids = tokenizer.encode(
    prompt,
    add_bos=True,
    add_eos=False,
)
print(f"\nPrompt token count: {len(prompt_ids)}")
# input_ids = torch.tensor(
#     [prompt_ids],
#     dtype=torch.long,
#     device=device,
# )



# ==========================================================
# Generate
# ==========================================================

with torch.no_grad():

    generated_ids = generator.generate(
        input_ids=prompt_ids,
        max_new_tokens=80,
        temperature=0.7,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
    )


# ==========================================================
# Decode
# ==========================================================

# generated_text = tokenizer.decode(
#     generated_ids[0].tolist()
# )
generated_text = tokenizer.decode(
    generated_ids
)


# ==========================================================
# Display
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING PROMPT")
print("=" * 70)

print(prompt)

print("\n" + "=" * 70)
print("EXPECTED TRAINING ANSWER")
print("=" * 70)

print(expected_answer)

print("\n" + "=" * 70)
print("GENIEE GENERATED OUTPUT")
print("=" * 70)

print(generated_text)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)