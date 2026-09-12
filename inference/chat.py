import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.geniee_model import GenieeModel  # Adjust import based on your model file structure
from tokenizer.tokenizer import GenieeTokenizer  # Adjust import based on your tokenizer script
from model.config import GenieeConfig

def generate_response(model, tokenizer, prompt, max_new_tokens=128, device="cpu"):
    model.eval()
    
    # Format input prompt
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long).to(device)
    prompt_length = input_ids.shape[1]
    
    eos_id = tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else 3

    with torch.no_grad():
        for _ in range(max_new_tokens):
            cond_input = input_ids[:, -256:]
            logits = model(cond_input)
            
            next_token_logits = logits[:, -1, :]
            temperature = 0.7
            top_k = 40

            logits = next_token_logits / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
           # next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token.item() == eos_id:
                break

            input_ids = torch.cat((input_ids, next_token), dim=1)

    # Slice ONLY the newly generated tokens (excluding prompt tokens)
    generated_tokens = input_ids[0][prompt_length:].tolist()
    
    # Decode only generated text
    raw_response = tokenizer.decode(generated_tokens)
    
    # Clean control characters and formatting noise
    clean_response = (
        raw_response
        .replace("⁇", "")
        .replace("im_start", "")
        .replace("im_end", "")
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .strip() 
    )
    
    return clean_response

def main():
    checkpoint_path = Path("checkpoints/geniee_sft_best.pt")
    tokenizer_path = Path(r"tokenizer/artifacts/geniee.model")
    device = "cpu"

    print("Loading tokenizer and model...")
    tokenizer = GenieeTokenizer(tokenizer_path)
    
    # Initialize model using your configuration
    config = GenieeConfig(
        vocab_size=2048,
        max_seq_len=256,
        d_model=512,
        num_heads=8,
        ffn_hidden_dim=2048,
        num_layers=8,
        dropout=0.1,
    )
    model = GenieeModel(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Support loading full checkpoint dict or state_dict directly
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    print("\nGENIEE AI CHAT READY (Type 'exit' to quit)\n" + "="*40)

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
                
            response = generate_response(model, tokenizer, user_input, device=device)
            print(f"\nGeniee: {response}")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()