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