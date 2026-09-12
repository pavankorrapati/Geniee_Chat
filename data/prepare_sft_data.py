import json
from pathlib import Path

def build_sft_dataset():
    raw_instructions = Path("corpus/instructions")
    output_file = Path("data/processed/sft_data.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sft_samples = []
    
    # Process text instruction files
    for file_path in raw_instructions.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
            
        # Parse user/assistant turns or prompt structures
        entries = content.split("\n\n")
        for entry in entries:
            if "Q:" in entry and "A:" in entry:
                parts = entry.split("A:")
                q_text = parts[0].replace("Q:", "").strip()
                a_text = parts[1].strip()
                
                # Format into chat instruction template
                formatted_text = f"<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n{a_text}<|im_end|>"
                sft_samples.append({"text": formatted_text})

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    build_sft_dataset()