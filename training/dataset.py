import json
import torch
from torch.utils.data import Dataset

class GenieeSFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, seq_len=256):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item["text"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Mask padding tokens in loss calculations
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, labels