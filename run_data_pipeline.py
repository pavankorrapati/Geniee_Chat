from pathlib import Path

from data.build_dataset import GenieeDataModule


PROJECT_ROOT = Path(__file__).resolve().parent


data_module = GenieeDataModule(

    corpus_dir=(
        PROJECT_ROOT / "corpus"
    ),

    tokenizer_path=(
        PROJECT_ROOT
        / "tokenizer"
        / "artifacts"
        / "geniee.model"
    ),

    max_seq_len=8,

    batch_size=2,

    stride=4,

    validation_split=0.1,

    shuffle_train=True
)


train_loader, validation_loader = (
    data_module.build()
)


data_module.summary()


# print()
# print("Getting one training batch...")


input_ids, target_ids = next(
    iter(train_loader)
)


# print()
# print("INPUT:")
# print(input_ids)


# print()
# print("TARGET:")
# print(target_ids)




# ==========================================================
# GENIEE - COMPLETE EXECUTION FLOW
# ==========================================================

# # 1. Check tokenizer
# python tokenizer\test_tokenizer.py

# # 2. Inspect corpus
# python data\inspect_corpus.py

# # 3. Build instruction/SFT dataset
# python data\build_instruction_dataset.py

# # 4. Verify pretrained checkpoint exists
# Test-Path checkpoints\geniee_pretrain_best.pt

# # 5. Train SFT from pretrained Geniee model
# python training\train_sft.py

# # 6. Evaluate SFT model
# python evaluation\evaluate_sft.py --greedy --limit 24

# # 7. Start Geniee chat
# python generation\chat.py