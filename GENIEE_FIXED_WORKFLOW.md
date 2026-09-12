# Geniee – fixed training and chat workflow

## Why the old chat was poor

The project had a few architectural mismatches rather than one sampling bug:

1. `generation/chat.py` loaded `geniee_epoch_5.pt`, a base-language-model checkpoint, instead of the SFT checkpoint.
2. The CLI built prompts as `User: ... / Geniee:`, while SFT training used `<|system|>`, `<|user|>`, `<|assistant|>`.
3. Generation stripped the trailing newline after `<|assistant|>`, while SFT trained with that newline.
4. SFT created a fresh model instead of initializing from the pretrained model, throwing away pretrained language knowledge.
5. The model context was only 128 tokens for chat; SFT now uses 256.
6. Generation could penalize words from the user's prompt instead of only newly generated tokens.
7. Special BOS/UNK tokens were not consistently protected in every generation path.
8. Some old tests and utility classes had drifted away from the current APIs.

## Correct order

### 1. Rebuild tokenizer only if corpus/tokenizer changed

```powershell
python tokenizer/train_tokenizer.py
```

The checked-in tokenizer currently reports vocabulary size 2048.

### 2. Build instruction data

```powershell
python data/build_instruction_dataset.py
```

### 3. Pretrain the base language model

```powershell
python training/train_pretrain.py
```

This creates:

`checkpoints/geniee_pretrain_best.pt`

### 4. Instruction-tune from the pretrained checkpoint

```powershell
python training/train_sft.py
```

The SFT trainer automatically loads `checkpoints/geniee_pretrain_best.pt` when it exists.

To intentionally train from random weights:

```powershell
python training/train_sft.py --from-scratch
```

### 5. Evaluate held-out questions

```powershell
python evaluation/evaluate_sft.py --greedy --limit 24
```

### 6. Start the chatbot

```powershell
python generation/chat.py
```

The chatbot now uses:

`checkpoints/geniee_sft_best.pt`

and the exact same role format used during SFT.

## Important expectation

This is still a small local model. The code fixes the training/inference pipeline, but model quality is bounded by the amount and quality of training data and compute. It should become substantially more consistent after the pretrained -> SFT pipeline is run correctly, but it should not be expected to match Gemini or ChatGPT from a small corpus and a ~27M parameter model.
