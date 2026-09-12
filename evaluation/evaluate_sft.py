from __future__ import annotations

from pathlib import Path
import json
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat.prompt import DEFAULT_SYSTEM_PROMPT, build_prompt
from generation.generate import load_geniee


def main():
    parser = argparse.ArgumentParser(description="Evaluate Geniee on held-out instruction examples.")
    parser.add_argument("--checkpoint", default="checkpoints/geniee_sft_best.pt")
    parser.add_argument("--split", default="test.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    data_path = PROJECT_ROOT / "data" / "processed" / "splits" / args.split
    generator = load_geniee(checkpoint)

    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = records[:args.limit]

    print("=" * 70)
    print("GENIEE SFT EVALUATION")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint}")
    print(f"Examples   : {len(records)}")

    for index, record in enumerate(records, 1):
        system = next((m["content"] for m in record["messages"] if m["role"] == "system"), DEFAULT_SYSTEM_PROMPT)
        user = next(m["content"] for m in record["messages"] if m["role"] == "user")
        expected = next(m["content"] for m in record["messages"] if m["role"] == "assistant")
        prompt = build_prompt(system, [{"role": "user", "content": user}])
        answer = generator.generate(
            prompt,
            max_new_tokens=80,
            temperature=0.65,
            top_k=30,
            top_p=0.90,
            repetition_penalty=1.05,
            do_sample=not args.greedy,
            return_full_text=False,
        )
        print("\n" + "-" * 70)
        print(f"Example {index}")
        print(f"Q: {user}")
        print(f"Expected: {expected}")
        print(f"Geniee  : {answer}")


if __name__ == "__main__":
    main()
