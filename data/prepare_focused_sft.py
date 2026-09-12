import json
from pathlib import Path

# Focus the instruction set on a single domain task
focused_dataset = [
    {"text": "<|im_start|>user\nConvert to Gherkin: User logs in with valid credentials<|im_end|>\n<|im_start|>assistant\nGiven the user is on the login page\nWhen they enter valid credentials\nThen they should be redirected to dashboard<|im_end|>"},
    # Add 500-1000 domain-specific pairs here
]

output = Path("data/processed/sft_focused.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)
with open(output, "w", encoding="utf-8") as f:
    for entry in focused_dataset:
        f.write(json.dumps(entry) + "\n")