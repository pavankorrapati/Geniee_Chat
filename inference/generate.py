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
# # Configuration
# # ==========================================================

# TOKENIZER_PATH = (
#     PROJECT_ROOT
#     / "tokenizer"
#     / "artifacts"
#     / "geniee.model"
# )

# CHECKPOINT_PATH = (
#     PROJECT_ROOT
#     / "checkpoints"
#     / "geniee_best.pt"
# )


# MAX_NEW_TOKENS = 50

# TEMPERATURE = 0.8


# # ==========================================================
# # Device
# # ==========================================================

# DEVICE = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )


# # ==========================================================
# # Load tokenizer
# # ==========================================================

# def load_tokenizer():

#     if not TOKENIZER_PATH.exists():

#         raise FileNotFoundError(
#             f"Tokenizer not found:\n"
#             f"{TOKENIZER_PATH}"
#         )

#     tokenizer = GenieeTokenizer(
#         model_path=TOKENIZER_PATH
#     )

#     return tokenizer


# # ==========================================================
# # Load model
# # ==========================================================

# def load_model(
#     tokenizer
# ):

#     if not CHECKPOINT_PATH.exists():

#         raise FileNotFoundError(
#             f"Checkpoint not found:\n"
#             f"{CHECKPOINT_PATH}"
#         )

#     config = GenieeConfig(

#         vocab_size=(
#             tokenizer.vocab_size
#         ),

#         max_seq_len=128
#     )

#     model = GenieeModel(
#         config
#     )

#     checkpoint = torch.load(
#         CHECKPOINT_PATH,
#         map_location=DEVICE
#     )

#     model.load_state_dict(
#         checkpoint[
#             "model_state_dict"
#         ]
#     )

#     model.to(
#         DEVICE
#     )

#     model.eval()

#     return model


# # ==========================================================
# # Generate text
# # ==========================================================

# # @torch.no_grad()
# # def generate(
# #     model,
# #     tokenizer,
# #     prompt,
# #     max_new_tokens=50,
# #     temperature=0.8
# # ):

# #     # ------------------------------------------------------
# #     # Encode prompt
# #     # ------------------------------------------------------

# #     token_ids = tokenizer.encode(
# #         prompt,
# #         add_bos=True,
# #         add_eos=False
# #     )

# #     input_ids = torch.tensor(
# #         [token_ids],
# #         dtype=torch.long,
# #         device=DEVICE
# #     )

# #     # ------------------------------------------------------
# #     # Generate
# #     # ------------------------------------------------------

# #     for _ in range(
# #         max_new_tokens
# #     ):

# #         # Keep context within model limit
# #         input_ids = input_ids[
# #             :,
# #             -128:
# #         ]

# #         # Forward pass
# #         logits = model(
# #             input_ids
# #         )

# #         # Last token prediction
# #         next_token_logits = logits[
# #             :, -1, :
# #         ]

# #         # Temperature
# #         next_token_logits = (
# #             next_token_logits
# #             / temperature
# #         )

# #         # Probability distribution
# #         probabilities = torch.softmax(
# #             next_token_logits,
# #             dim=-1
# #         )

# #         # Sample next token
# #         next_token = torch.multinomial(
# #             probabilities,
# #             num_samples=1
# #         )

# #         # Append token
# #         input_ids = torch.cat(
# #             [
# #                 input_ids,
# #                 next_token
# #             ],
# #             dim=1
# #         )

# #         # EOS
# #         # if next_token.item() == 3:
# #         #     break
# #         if next_token.item() == tokenizer.eos_id:
# #             break

# #     # ------------------------------------------------------
# #     # Decode
# #     # ------------------------------------------------------

# #     generated_ids = (
# #         input_ids[0]
# #         .detach()
# #         .cpu()
# #         .tolist()
# #     )

# #     return tokenizer.decode(
# #         generated_ids
# #     )
# # @torch.no_grad()
# # def generate(
# #     model,
# #     tokenizer,
# #     prompt,
# #     max_new_tokens=50
# # ):

# #     # ------------------------------------------------------
# #     # Encode prompt
# #     # ------------------------------------------------------

# #     token_ids = tokenizer.encode(
# #         prompt,
# #         add_bos=True,
# #         add_eos=False
# #     )

# #     input_ids = torch.tensor(
# #         [token_ids],
# #         dtype=torch.long,
# #         device=DEVICE
# #     )

# #     # ------------------------------------------------------
# #     # Generate
# #     # ------------------------------------------------------

# #     for _ in range(max_new_tokens):

# #         # Keep context within model limit
# #         input_ids = input_ids[
# #             :,
# #             -128:
# #         ]

# #         # Forward pass
# #         logits = model(
# #             input_ids
# #         )

# #         # Last token prediction
# #         next_token_logits = logits[
# #             :, -1, :
# #         ]

# #         # --------------------------------------------------
# #         # Greedy decoding
# #         # --------------------------------------------------

# #         next_token = torch.argmax(
# #             next_token_logits,
# #             dim=-1,
# #             keepdim=True
# #         )

# #         # --------------------------------------------------
# #         # Append token
# #         # --------------------------------------------------

# #         input_ids = torch.cat(
# #             [
# #                 input_ids,
# #                 next_token
# #             ],
# #             dim=1
# #         )

# #         # --------------------------------------------------
# #         # EOS
# #         # --------------------------------------------------

# #         if next_token.item() == tokenizer.eos_id:
# #             break

# #     # ------------------------------------------------------
# #     # Decode
# #     # ------------------------------------------------------

# #     generated_ids = (
# #         input_ids[0]
# #         .detach()
# #         .cpu()
# #         .tolist()
# #     )

# #     return tokenizer.decode(
# #         generated_ids
# #     )

# # @torch.no_grad()
# # def generate(
# #     model,
# #     tokenizer,
# #     prompt,
# #     max_new_tokens=50
# # ):

# #     # --------------------------------------------------
# #     # Format prompt as an instruction conversation
# #     # --------------------------------------------------

# #     formatted_prompt = (
# #         "User: "
# #         + prompt.strip()
# #         + "\nAssistant:"
# #     )

# #     # --------------------------------------------------
# #     # Encode prompt
# #     # --------------------------------------------------

# #     token_ids = tokenizer.encode(
# #         formatted_prompt,
# #         add_bos=True,
# #         add_eos=False
# #     )

# #     input_ids = torch.tensor(
# #         [token_ids],
# #         dtype=torch.long,
# #         device=DEVICE
# #     )

# #     prompt_length = input_ids.shape[1]

# #     # --------------------------------------------------
# #     # Generate
# #     # --------------------------------------------------

# #     for _ in range(max_new_tokens):

# #         # Keep context within model limit
# #         model_input = input_ids[
# #             :,
# #             -128:
# #         ]

# #         # Forward pass
# #         logits = model(
# #             model_input
# #         )

# #         # Last token
# #         next_token_logits = logits[
# #             :,
# #             -1,
# #             :
# #         ]

# #         # Greedy decoding
# #         next_token = torch.argmax(
# #             next_token_logits,
# #             dim=-1,
# #             keepdim=True
# #         )

# #         # Append
# #         input_ids = torch.cat(
# #             [
# #                 input_ids,
# #                 next_token
# #             ],
# #             dim=1
# #         )

# #         # Stop at EOS
# #         if (
# #             next_token.item()
# #             == tokenizer.eos_id
# #         ):
# #             break

# #     # --------------------------------------------------
# #     # Only return generated tokens
# #     # --------------------------------------------------

# #     generated_ids = (
# #         input_ids[
# #             0,
# #             prompt_length:
# #         ]
# #         .detach()
# #         .cpu()
# #         .tolist()
# #     )

# #     # --------------------------------------------------
# #     # Decode
# #     # --------------------------------------------------

# #     output = tokenizer.decode(
# #         generated_ids
# #     )

# #     return output.strip()

# # # ==========================================================
# # # Main
# # # ==========================================================

# # def main():

# #     print()
# #     print("=" * 60)
# #     print("GENIEE TEXT GENERATION")
# #     print("=" * 60)

# #     print(
# #         f"Device: {DEVICE}"
# #     )

# #     tokenizer = load_tokenizer()

# #     model = load_model(
# #         tokenizer
# #     )

# #     print(
# #         f"Vocabulary size: "
# #         f"{tokenizer.vocab_size}"
# #     )

# #     print()
# #     print(
# #         "Model loaded successfully."
# #     )

# #     print()
# #     print(
# #         "Enter a prompt."
# #     )

# #     print(
# #         "Type 'exit' to quit."
# #     )

# #     print()

# #     while True:

# #         prompt = input(
# #             "You: "
# #         ).strip()

# #         if prompt.lower() == "exit":
# #             break

# #         if not prompt:
# #             continue

# #         output = generate(model=model,tokenizer=tokenizer,prompt=prompt,max_new_tokens=MAX_NEW_TOKENS)

# #         print()
# #         print(
# #             "Geniee:",
# #             output
# #         )

# #         print()


# # # ==========================================================
# # # Entry point
# # # ==========================================================

# # if __name__ == "__main__":

# #     main()

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
#     sys.path.insert(0, str(PROJECT_ROOT))


# # ==========================================================
# # Imports
# # ==========================================================

# from tokenizer.tokenizer import GenieeTokenizer
# from model.config import GenieeConfig
# from model.model import GenieeModel


# # ==========================================================
# # Paths
# # ==========================================================

# TOKENIZER_PATH = (
#     PROJECT_ROOT
#     / "tokenizer"
#     / "artifacts"
#     / "geniee.model"
# )

# CHECKPOINT_PATH = (
#     PROJECT_ROOT
#     / "checkpoints"
#     / "geniee_best.pt"
# )


# # ==========================================================
# # Device
# # ==========================================================

# DEVICE = torch.device("cpu")


# # ==========================================================
# # Load tokenizer
# # ==========================================================

# tokenizer = GenieeTokenizer(
#     model_path=TOKENIZER_PATH
# )


# # ==========================================================
# # Model configuration
# # ==========================================================

# config = GenieeConfig(
#     vocab_size=tokenizer.vocab_size,
#     max_seq_len=128,
#     d_model=512,
#     num_heads=8,
#     ffn_hidden_dim=2048,
#     num_layers=8,
#     dropout=0.1
# )


# # ==========================================================
# # Build model
# # ==========================================================

# model = GenieeModel(config)

# checkpoint = torch.load(
#     CHECKPOINT_PATH,
#     map_location=DEVICE
# )

# # Support checkpoints containing either
# # model_state_dict or state_dict.
# if "model_state_dict" in checkpoint:
#     state_dict = checkpoint["model_state_dict"]
# elif "state_dict" in checkpoint:
#     state_dict = checkpoint["state_dict"]
# else:
#     state_dict = checkpoint

# model.load_state_dict(state_dict)

# model.to(DEVICE)
# model.eval()


# # ==========================================================
# # Generation
# # ==========================================================

# @torch.no_grad()
# def generate(
#     prompt,
#     max_new_tokens=50
# ):
#     """
#     Generate an answer for a user prompt.

#     The prompt is formatted exactly like
#     the training data.
#     """

#     formatted_prompt = (
#         "User: "
#         + prompt.strip()
#         + "\nAssistant:"
#     )

#     token_ids = tokenizer.encode(
#         formatted_prompt,
#         add_bos=True,
#         add_eos=False
#     )

#     input_ids = torch.tensor(
#         [token_ids],
#         dtype=torch.long,
#         device=DEVICE
#     )

#     prompt_length = input_ids.shape[1]

#     for _ in range(max_new_tokens):

#         # Keep only the most recent context.
#         model_input = input_ids[:, -128:]

#         logits = model(model_input)

#         next_token_logits = logits[:, -1, :]

#         # Greedy decoding for baseline testing.
#         next_token = torch.argmax(
#             next_token_logits,
#             dim=-1,
#             keepdim=True
#         )

#         input_ids = torch.cat(
#             [input_ids, next_token],
#             dim=1
#         )

#         if next_token.item() == tokenizer.eos_id:
#             break

#     # Only decode tokens generated after
#     # the original prompt.
#     generated_ids = (
#         input_ids[
#             0,
#             prompt_length:
#         ]
#         .detach()
#         .cpu()
#         .tolist()
#     )

#     output = tokenizer.decode(
#         generated_ids
#     )

#     return output.strip()


# # ==========================================================
# # Interactive CLI
# # ==========================================================

# if __name__ == "__main__":

#     print()
#     print("=" * 60)
#     print("GENIEE INFERENCE")
#     print("=" * 60)

#     print(f"Tokenizer : {TOKENIZER_PATH}")
#     print(f"Checkpoint: {CHECKPOINT_PATH}")
#     print(f"Device    : {DEVICE}")

#     print()
#     print("Type 'exit' to quit.")
#     print()

#     while True:

#         prompt = input("User: ").strip()

#         if prompt.lower() == "exit":
#             break

#         if not prompt:
#             continue

#         response = generate(
#             prompt,
#             max_new_tokens=50
#         )

#         print()
#         print("Geniee:", response)
#         print()

# from pathlib import Path
# import sys

# import torch


# ==========================================================
# Project root
# ==========================================================


from __future__ import annotations

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
    / "geniee_pretrain_v2_best.pt"
)


# ==========================================================
# Configuration
# ==========================================================

MAX_SEQ_LEN = 128

D_MODEL = 512

NUM_HEADS = 8

FFN_HIDDEN_DIM = 2048

NUM_LAYERS = 8

DROPOUT = 0.1


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# Load model
# ==========================================================

def load_model(
    tokenizer,
):

    config = GenieeConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    model = GenieeModel(
        config=config
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ==========================================================
# Build prompt
# ==========================================================

def build_prompt(
    system_message,
    user_message,
):
    """
    Build exactly the same conversation format
    used during SFT.
    """

    return (
        "<|system|>\n"
        + system_message.strip()
        + "\n"
        + "<|user|>\n"
        + user_message.strip()
        + "\n"
        + "<|assistant|>\n"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 70)
    print(
        "GENIEE V2 INFERENCE TEST"
    )
    print("=" * 70)

    print()
    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Tokenizer  : {TOKENIZER_PATH}"
    )

    print(
        f"Checkpoint : {CHECKPOINT_PATH}"
    )

    # ------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------

    tokenizer = GenieeTokenizer(
        model_path=TOKENIZER_PATH
    )

    print(
        f"Vocabulary : {tokenizer.vocab_size}"
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    print()
    print(
        "Loading Geniee model..."
    )

    model = load_model(
        tokenizer
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters : "
        f"{parameter_count:,}"
    )

    print(
        "Model loaded successfully."
    )

    # ------------------------------------------------------
    # Generator
    # ------------------------------------------------------

    generator = GenieeGenerator(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
    )

    # ------------------------------------------------------
    # Test prompt
    # ------------------------------------------------------

    system_message = (
        "You are Geniee, an AI assistant "
        "specialized in software testing "
        "and test automation."
    )

    user_message = (
        "What is Selenium?"
    )

    prompt = build_prompt(
        system_message=system_message,
        user_message=user_message,
    )

    print()
    print("-" * 70)
    print("PROMPT")
    print("-" * 70)

    print(prompt)

    # ------------------------------------------------------
    # Encode
    # ------------------------------------------------------

    prompt_ids = tokenizer.encode(
        prompt,
        add_bos=True,
        add_eos=False,
    )

    print(
        f"Prompt tokens : "
        f"{len(prompt_ids)}"
    )

    if len(prompt_ids) >= MAX_SEQ_LEN:

        raise RuntimeError(
            "Prompt is too long for the "
            "current 128-token context."
        )

    # ------------------------------------------------------
    # Generate
    # ------------------------------------------------------

    print()
    print("-" * 70)
    print("GENERATING")
    print("-" * 70)

    # generated_text = (
    #     generator.generate_text(
    #         input_ids=prompt_ids,
    #         max_new_tokens=80,
    #         temperature=0.8,
    #         top_k=40,
    #         top_p=0.90,
    #         repetition_penalty=1.1,
    #     )
    # )
    generated_text = (
        generator.generate_text(
            input_ids=prompt_ids,
            max_new_tokens=60,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=False,
        )
    )

    # ------------------------------------------------------
    # Extract assistant response
    # ------------------------------------------------------

    assistant_marker = (
        "<|assistant|>\n"
    )

    if assistant_marker in generated_text:

        response = generated_text.split(
            assistant_marker,
            1
        )[1]

    else:

        response = generated_text

    # Remove obvious EOS artifacts if any.

    response = response.strip()

    print()
    print("-" * 70)
    print("GENIEE RESPONSE")
    print("-" * 70)

    print(response)

    print()
    print("=" * 70)
    print(
        "INFERENCE TEST COMPLETE"
    )
    print("=" * 70)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()

# ==========================================================
# Imports
# ==========================================================

# from tokenizer.tokenizer import GenieeTokenizer
# from model.config import GenieeConfig
# from model.geniee_model import GenieeModel


# # ==========================================================
# # Paths
# # ==========================================================

# TOKENIZER_PATH = (
#     PROJECT_ROOT
#     / "tokenizer"
#     / "artifacts"
#     / "geniee.model"
# )

# CHECKPOINT_PATH = (
#     PROJECT_ROOT
#     / "checkpoints"
#     / "geniee_best.pt"
# )


# # ==========================================================
# # Device
# # ==========================================================

# DEVICE = torch.device("cpu")


# # ==========================================================
# # Load tokenizer
# # ==========================================================

# tokenizer = GenieeTokenizer(
#     model_path=TOKENIZER_PATH
# )


# # ==========================================================
# # Model configuration
# # ==========================================================

# config = GenieeConfig(
#     vocab_size=tokenizer.vocab_size,
#     max_seq_len=128,
#     d_model=512,
#     num_heads=8,
#     ffn_hidden_dim=2048,
#     num_layers=8,
#     dropout=0.1
# )


# # ==========================================================
# # Build model
# # ==========================================================

# model = GenieeModel(config)

# checkpoint = torch.load(
#     CHECKPOINT_PATH,
#     map_location=DEVICE
# )

# # Support checkpoints containing either
# # model_state_dict or state_dict.
# if "model_state_dict" in checkpoint:
#     state_dict = checkpoint["model_state_dict"]
# elif "state_dict" in checkpoint:
#     state_dict = checkpoint["state_dict"]
# else:
#     state_dict = checkpoint

# model.load_state_dict(state_dict)

# model.to(DEVICE)
# model.eval()


# # ==========================================================
# # Generation
# # ==========================================================

# @torch.no_grad()
# def generate(
#     prompt,
#     max_new_tokens=50
# ):
#     """
#     Generate an answer for a user prompt.

#     The prompt is formatted exactly like
#     the training data.
#     """

#     formatted_prompt = (
#         "User: "
#         + prompt.strip()
#         + "\nAssistant:"
#     )

#     token_ids = tokenizer.encode(
#         formatted_prompt,
#         add_bos=True,
#         add_eos=False
#     )

#     input_ids = torch.tensor(
#         [token_ids],
#         dtype=torch.long,
#         device=DEVICE
#     )

#     prompt_length = input_ids.shape[1]

#     for _ in range(max_new_tokens):

#         # Keep only the most recent context.
#         model_input = input_ids[:, -128:]

#         logits = model(model_input)

#         next_token_logits = logits[:, -1, :]

#         # Greedy decoding for baseline testing.
#         next_token = torch.argmax(
#             next_token_logits,
#             dim=-1,
#             keepdim=True
#         )

#         input_ids = torch.cat(
#             [input_ids, next_token],
#             dim=1
#         )

#         if next_token.item() == tokenizer.eos_id:
#             break

#     # Only decode tokens generated after
#     # the original prompt.
#     generated_ids = (
#         input_ids[
#             0,
#             prompt_length:
#         ]
#         .detach()
#         .cpu()
#         .tolist()
#     )

#     output = tokenizer.decode(
#         generated_ids
#     )

#     return output.strip()


# # ==========================================================
# # Interactive CLI
# # ==========================================================

# if __name__ == "__main__":

#     print()
#     print("=" * 60)
#     print("GENIEE INFERENCE")
#     print("=" * 60)

#     print(f"Tokenizer : {TOKENIZER_PATH}")
#     print(f"Checkpoint: {CHECKPOINT_PATH}")
#     print(f"Device    : {DEVICE}")

#     print()
#     print("Type 'exit' to quit.")
#     print()

#     while True:

#         prompt = input("User: ").strip()

#         if prompt.lower() == "exit":
#             break

#         if not prompt:
#             continue

#         response = generate(
#             prompt,
#             max_new_tokens=50
#         )

#         print()
#         print("Geniee:", response)
#         print()